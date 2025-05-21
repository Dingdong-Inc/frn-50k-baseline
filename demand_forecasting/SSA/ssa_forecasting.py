import pandas as pd
import numpy as np
from datasets import load_dataset
import argparse

def ssa_predict(demand_df):
    dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
    eval_df = dataset['eval'].to_pandas()
    all_data = pd.concat([demand_df, eval_df], axis=0)
    all_data = all_data.sort_values(by=['store_id', 'product_id', 'dt'])
    # calendar feature
    all_data['time_idx'] = (pd.to_datetime(all_data['dt']) - pd.to_datetime(all_data['dt']).min()).dt.days
    all_data['dow'] = pd.to_datetime(all_data['dt']).dt.dayofweek
    all_data.loc[all_data['holiday_flag']==1, 'dow'] = 5
    all_data.loc[(all_data['holiday_flag']==0)&(all_data['dow'].isin([5,6])), 'dow'] = 0
    # precipitation feature
    all_data['is_precpt'] = (all_data['precpt']>3.5).astype(int)
    # continuouse and catigorical variables
    x_cont = all_data[['time_idx', 'discount', 'sale_amount', 'sale_amount_pred']].values.reshape(-1, 97, 4)
    x_cat = all_data[['dow', 'holiday_flag', 'is_precpt']].values.reshape(-1, 97, 3)
    # config
    max_encoder_length=90
    time_idx=0
    discount_idx=1
    dow_idx=0
    holiday_idx=1
    is_precpt_idx=2
    holiday_label = 5
    # weight
    date_distance = 18 - x_cont[:, :max_encoder_length, time_idx][:,None,:]/5
    discount_distance = 1 - np.abs((x_cont[:, max_encoder_length:, discount_idx][...,None] - x_cont[:, :max_encoder_length, discount_idx][:,None,:]))
    decoder_dow = (x_cat[:, max_encoder_length:, dow_idx][...,None])
    encoder_dow = (x_cat[:, :max_encoder_length, dow_idx][:,None,:])
    dow_distance = decoder_dow - encoder_dow
    decoder_precpt = (x_cat[:, max_encoder_length:, is_precpt_idx][...,None])
    encoder_precpt = (x_cat[:, :max_encoder_length, is_precpt_idx][:,None,:])
    precpt_distance = decoder_precpt - encoder_precpt
    date_distance = np.where((decoder_dow == holiday_label).astype(int) - (encoder_dow == holiday_label).astype(int) == 0, date_distance - 3, date_distance)
    date_distance = np.where((decoder_dow != holiday_label) & (encoder_dow == holiday_label), date_distance + 3, date_distance)
    date_distance = np.where(dow_distance == 0, date_distance - 3, date_distance)
    date_distance = np.where((precpt_distance == 0) & (decoder_precpt == 1), date_distance - 3, date_distance)
    date_distance = np.where(precpt_distance != 0, date_distance + 3, date_distance)
    date_distance = np.clip(date_distance, a_min=1, a_max=18)
    weight = (18-date_distance) * discount_distance**2
    weight = np.exp(weight)/np.exp(weight).sum(axis=-1, keepdims=True)

    # high sale & low sale info
    mu = all_data.query("dt<='2024-06-25'").groupby(['store_id', 'product_id']).agg({'sale_amount':'mean'})
    mu = mu.reset_index().rename(columns={'sale_amount':'psd'})
    all_data = all_data.merge(mu, on=['store_id', 'product_id'], how='left')

    pdf = all_data.query("dt>='2024-06-26'").copy()
    target_cols = ['sale_amount', 'sale_amount_pred']
    for target_col in target_cols:
        target_idx = target_cols.index(target_col)+2
        target = x_cont[:, :max_encoder_length, target_idx:target_idx+1]
        index = encoder_dow.squeeze()[...,None] == np.arange(6)
        mean_sale = np.nanmean(np.where(index, target, np.nan), axis=1) + 0.001
        season_ratio = np.nanmedian(mean_sale[...,None] / mean_sale[:,None,...], axis=0)
        ratio = season_ratio[decoder_dow, encoder_dow]
        pred = (weight * ratio)@target
        pdf[f'ssa_pred_{target_col}'] = pred.reshape(-1)
    # overall
    metric = pd.concat([
        evaluation(pdf, 'psd>=0'), # overall
        evaluation(pdf, 'psd<1'), # low sale
        evaluation(pdf, 'psd>=1'), # high sale
    ], axis=0)
    print(metric)

def evaluation(pdf, condition='psd>=0'):
    res = []
    for pred_col in ['ssa_pred_sale_amount', 'ssa_pred_sale_amount_pred']:
        wape_list,mae_list,wpe_list = [],[],[]
        for target_dt, subdf in pdf.query('stock_hour6_22_cnt==0').query(condition).groupby('dt'):
            mae = (subdf['sale_amount'] - subdf[pred_col]).abs().sum()
            wape = mae / subdf['sale_amount'].sum()
            wape_list.append(wape)
            mae_list.append((subdf['sale_amount'] - subdf[pred_col]).abs().mean())
            wpe_list.append((subdf[pred_col]-subdf['sale_amount']).sum()/subdf['sale_amount'].sum())
        res.append(pd.DataFrame({'demand':[pred_col], 'wape':[round(np.mean(wape_list),4)], 'wpe':[round(np.mean(wpe_list),4)], 'mae':[round(np.mean(mae_list),4)]}))
    metric = pd.concat(res)
    metric['group'] = condition
    metric = metric[['group', 'demand', 'wape', 'wpe', 'mae']]
    return metric

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demand_path", 
        type=str, 
        default='../../latent_demand_recovery/exp/demand/demand.parquet',
        help="demand data path, default '../../latent_demand_recovery/exp/demand/demand.parquet'"
    )
    args = parser.parse_args()
    demand_df = pd.read_parquet(args.demand_path)
    ssa_predict(demand_df)