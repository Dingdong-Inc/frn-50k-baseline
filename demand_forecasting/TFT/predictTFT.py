import os
import configs.tft_config as config
import numpy as np
import pandas as pd
import torch
from trainer.model import Model
from dataset.dataset import Dataset
from models.tft.model import TemporalFusionTransformer
import time
from datetime import datetime, timedelta
from datasets import load_dataset

config.date = '2024-06-26'
config.quantiles = 7
config.use_gpu = True
config.num_workers = 32
config.dataset_config["max_prediction_length"] = 7
config.dataset_config["max_encoder_length"] = 70
target_dates = pd.date_range(start = config.date, periods = config.dataset_config["max_prediction_length"])
target_dates = [date.strftime("%Y-%m-%d") for date in target_dates]

def loadDataset():
    t0 = time.time()
    dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
    df_train = dataset['train'].to_pandas()
    df_eval = dataset['eval'].to_pandas()
    df = pd.concat([df_train, df_eval], ignore_index=True)
    df['day_of_week'] = df['dt'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d').weekday())
    df.loc[df.dt >= config.date, 'sale_amount'] = np.nan
    dataset = Dataset(df, config)
    t1 = time.time()
    print(f"dataset generation cost {t1-t0}")

    df_train_psd = df_train.groupby(config.dataset_config['group_ids'])['sale_amount'].mean().to_frame('psd') # average daily sales amount
    df_eval = pd.merge(df_eval, df_train_psd, on = config.dataset_config['group_ids'])
    return dataset, df_train, df_eval

def do_predict(dataset, model_path, df_eval, verbose=False):
    best_tft = TemporalFusionTransformer.load_from_checkpoint(model_path).cuda()

    t1 = time.time()
    predictions, index = best_tft.predict(dataset.predict_df, return_index=True)
    predictions[predictions < 0] = 0
    t2 = time.time()
    print(f"predict cost {t2-t1}") if verbose else None

    columns = target_dates + config.dataset_config["group_ids"]
    idx = np.array(index[config.dataset_config['group_ids']])
    preds = np.concatenate((predictions, idx), axis=1)
    preds = pd.DataFrame(preds, columns=columns)
    preds = preds.melt(id_vars=config.dataset_config['group_ids'],
                       value_vars=target_dates, var_name='dt', value_name='prediction')
    preds[['store_id', 'product_id']] = preds[['store_id', 'product_id']].astype(int)

    df_groudtruth = df_eval[['store_id', 'product_id', 'dt', 'sale_amount', 'stock_hour6_22_cnt', 'psd']]
    merged_df = pd.merge(df_groudtruth, preds, on = ['store_id', 'product_id', 'dt'])
    print(len(merged_df), len(merged_df) / config.dataset_config["max_prediction_length"]) if verbose else None

    return merged_df

def cal_metrics(df, data_type, groups=["psd>=0"]):
    sample_cnt, preds_sum, actuals_sum, preds_mean, actuals_mean  = {}, {}, {}, {}, {}
    acc, wape, wpe, mae, bias = {}, {}, {}, {}, {}
    res = pd.DataFrame()
    for group in groups:
        df1 = df.query(group)
        sub_res = pd.DataFrame(target_dates, columns=["date"])
        for target_date in target_dates:
            df2 = df1[(df1.dt == target_date) & (df1.stock_hour6_22_cnt == 0)]
            preds = df2.prediction
            actuals = df2.sale_amount
            sample_cnt[target_date] = len(df2)
            preds_sum[target_date] = preds.sum(axis=0)
            actuals_sum[target_date] = actuals.sum(axis=0)
            preds_mean[target_date] = preds.mean(axis=0)
            actuals_mean[target_date] = actuals.mean(axis=0)
            acc[target_date] = 1 - (preds - actuals).abs().sum(axis=0) / actuals.abs().sum(axis=0)
            wape[target_date] = (preds - actuals).abs().sum(axis=0) / actuals.abs().sum(axis=0)
            wpe[target_date] = (preds - actuals).sum(axis=0) / actuals.abs().sum(axis=0)
            mae[target_date] = (preds - actuals).abs().mean(axis=0)
            bias[target_date] = (preds - actuals).mean(axis=0)
        sub_res['sample_cnt'] = sample_cnt.values()
        sub_res['preds_sum'] = preds_sum.values()
        sub_res['actuals_sum'] = actuals_sum.values()
        sub_res['preds_mean'] = preds_mean.values()
        sub_res['actuals_mean'] = actuals_mean.values()
        sub_res['acc'] = acc.values()
        sub_res['wape'] = wape.values()
        sub_res['wpe'] = wpe.values()
        sub_res['mae'] = mae.values()
        sub_res['bias'] = bias.values()
        sub_res.insert(0, 'data_type', data_type)
        sub_res.insert(0, 'group_type', group)
        res = pd.concat([res, sub_res], ignore_index=True)
    return res

def get_sorted_files(directory='.'):
    files = [
        f for f in os.listdir(directory)
        if os.path.isfile(os.path.join(directory, f))
    ]
    sorted_files = sorted(files)
    return sorted_files

def get_pred(dataset, df_eval, v_num, data_type, start_epoch=0, verbose=False):
    model_dir = f"./lightning_logs/version_{v_num}/checkpoints"
    files = get_sorted_files(model_dir)
    df_pred = []
    for idx, file in enumerate(files):
        epoch_num = file.split('-')[0].split('=')[1]
        if start_epoch == -1:
            if idx + 1 == len(files):
                _data_type = f"{data_type}_{v_num}_{epoch_num}"
                print(_data_type)
                df_pred.append((_data_type, do_predict(dataset, os.path.join(model_dir, file), df_eval, verbose)))
        else:
            if epoch_num >= start_epoch:
                _data_type = f"{data_type}_{v_num}_{epoch_num}"
                print(_data_type)
                df_pred.append((_data_type, do_predict(dataset, os.path.join(model_dir, file), df_eval, verbose)))
    return df_pred

def get_metrics(df_pred, groups=["psd>=0"]):
    df_res = pd.DataFrame()
    for (data_type, df) in df_pred:
        df_ = cal_metrics(df, data_type, groups)
        df_res = pd.concat([df_res, df_], axis=0, ignore_index=True)
    return df_res

dataset_raw, df_train, df_eval = loadDataset()
df_pred = get_pred(dataset_raw, df_eval, v_num=0, data_type='raw', start_epoch=-1, verbose=True)
df_res = get_metrics(df_pred, ["psd>=0", "psd>=1", "psd<1"])
df_res.to_parquet("./raw_res_v0_last_epoch.parquet")

print(df_res.groupby(['group_type', 'data_type']).mean(numeric_only=True))