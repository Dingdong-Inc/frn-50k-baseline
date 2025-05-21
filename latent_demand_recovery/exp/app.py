import os, subprocess, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
sys.path.append(project_root)

import torch
from torch import nn
import pypots
from data import load_data
from model import load_model
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import logging
logging.basicConfig(level=logging.ERROR)
from datetime import datetime
from datasets import load_dataset
import argparse

def set_seed(seed_value=1024):
    np.random.seed(seed_value)
    random.seed(seed_value)
    # 设置CPU的种子
    torch.manual_seed(seed_value)
    # 如果你使用的是CUDA，还需要设置CUDA的种子
    torch.cuda.manual_seed(seed_value)
    # 如果你使用的是多GPU，还需要设置随机种子的 all-gather 方法
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    return


def _imputation(CONFIG):
    # load data
    data = load_data(CONFIG)
    (
        train_set, 
        ts_origin, 
        valid_idx,
    ) = (
        data['train_set'],
        data['ts_origin'], 
        data['valid_idx'],
    )
    # update specific dataset params config
    CONFIG.update(data['params'])
    model = load_model(CONFIG)
    model.fit(train_set)
    results = model.predict(train_set)
    if len(results['imputation'].shape) == 4:
        imputation = results['imputation'].mean(axis=1)[:,:,:CONFIG['OT']]
    else:
        imputation = results['imputation'][:,:,:CONFIG['OT']]
    imputation = np.where(imputation>0, imputation, 0)
    model_name = CONFIG['model']
    missing_rate = CONFIG['missing_rate']
    np.save(f'./demand/{model_name}_imputation_{missing_rate}.npy', imputation)
    if CONFIG['missing_rate']>0:
        evaluation_mnar(train_set['X'], imputation)
    return imputation
    
def _demand_recovery(imputation):
    dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
    data = dataset['train'].to_pandas()
    data = data.sort_values(by=['store_id', 'product_id', 'dt'])
    horizon=90
    series_num = data.shape[0]//horizon
    hours_sale = np.array(data['hours_sale'].tolist())
    hours_sale_origin = hours_sale.reshape(series_num*3, 30, 24)
    hours_sale_origin[...,6:22] = imputation.reshape(-1, 30, 16)
    sale_amount_pred = hours_sale_origin.sum(axis=-1).reshape(-1, 90)
    data[f'sale_amount_pred'] = sale_amount_pred.reshape(-1)
    data.to_parquet(f'./demand/demand.parquet')
    return data
    
def demand_recovery(CONFIG):
    imputation = _imputation(CONFIG)
    demand_df = None
    if CONFIG['missing_rate'] == 0:
        demand_df = _demand_recovery(imputation)
        evaluation_decoupling(demand_df)
    return demand_df


def evaluation_mnar(X, imputation):
    dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
    data = dataset['train'].to_pandas()
    data = data.sort_values(by=['store_id', 'product_id', 'dt'])
    horizon=90
    series_num = data.shape[0]//horizon
    hours_sale = np.array(data['hours_sale'].tolist())
    hours_stock_status = np.array(data['hours_stock_status'].tolist())

    hours_sale_origin = hours_sale.reshape(series_num*3, 30, 24)

    stock_hour_X = np.isnan(X[...,0].reshape(-1,30,16)).sum(axis=-1).reshape(-1,90).reshape(-1)
    stock_hour_origin = hours_stock_status[:,6:22].sum(axis=1)
    hours_sale_impute = hours_sale_origin.copy()
    hours_sale_impute[...,6:22] = imputation.reshape(-1, 30, 16)
    sale_amount_pred = hours_sale_impute.sum(axis=-1).reshape(-1, 90).reshape(-1)
    valid_idx = (stock_hour_X>0)&(stock_hour_origin==0) #&(data['mu']>=1)
    sale_amount = data['sale_amount'].values
    print('wape', (np.abs(sale_amount_pred-sale_amount)*valid_idx).sum()/(sale_amount*valid_idx).sum())
    print('wpe', ((sale_amount_pred-sale_amount)*valid_idx).sum()/(sale_amount*valid_idx).sum())
    
def evaluation_decoupling(data):
    df = data[['city_id', 'store_id', 'product_id', 'dt', 'holiday_flag', 'discount', 'sale_amount', 'sale_amount_pred','stock_hour6_22_cnt']].copy()
    mu = df.query('stock_hour6_22_cnt==0').groupby(['store_id', 'product_id'])['sale_amount'].mean()
    mu = mu.reset_index().rename(columns={'sale_amount':'mu'})
    corr = df.query('stock_hour6_22_cnt>0').groupby(['store_id', 'product_id', 'holiday_flag']).apply(lambda subdf:subdf[['stock_hour6_22_cnt', 'sale_amount', 'sale_amount_pred']].corr().iloc[:1, 1:])
    stock_nunique = df.query('stock_hour6_22_cnt>0').groupby(['store_id', 'product_id', 'holiday_flag']).agg({'stock_hour6_22_cnt':'nunique'}).reset_index()
    stock_nunique = stock_nunique.rename(columns={'stock_hour6_22_cnt':'nunique'})
    corr = corr.reset_index().merge(mu, on=['store_id', 'product_id']).merge(stock_nunique.query('nunique>3'), on=['store_id', 'product_id', 'holiday_flag'])
    metric = pd.DataFrame({
            'method':['sale_amount', 'sale_amount_pred'],
            'decouling score':np.nansum(corr[['sale_amount', 'sale_amount_pred']].values * corr[['mu']].values, axis=0)/corr['mu'].sum()
             })
    print(metric)
    
    
# default params config
CONFIG = {
    'model': 'DLinear',
    'saving_path': './save',
    'EPOCHS': 5,
    'batch_size': 128,
    'patience': 5,
    'n_layers': 2,
    'd_model': 64,
    'd_ffn': 32,
    'n_heads': 4,
    'd_k': 16,
    'd_v': 16,
    'dropout': 0.,
    'attn_dropout': 0.,
    'lr': 0.001, 
    'weight_decay': 1e-5, 
    'OT': 1,
    'missing_rate':0.3,
    'n_patches':7,
    'alpha': 1e-2
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", 
        type=str, 
        default='TimesNet',
        help="Demand Recovery Model, default='TimesNet'"
    )
    parser.add_argument(
        "--missing_rate", 
        type=float, 
        default=0.,
        help="Missing Rate for Artificial MNAR Evaluation, default missing_rate = 0 for latent demand recovery"
    )
    args = parser.parse_args()
    print(args)
    CONFIG['model'] = args.model
    CONFIG['missing_rate'] = args.missing_rate
    ## set random seed
    set_seed(seed_value=1024)
    ## latent demand recovery
    demand_df = demand_recovery(CONFIG)