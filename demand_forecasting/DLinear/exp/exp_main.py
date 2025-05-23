from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import dlinear
from utils.tools import EarlyStopping, adjust_learning_rate, visual, test_params_flop
from utils.metrics import metric
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from datasets import load_dataset
from torch import optim
from tqdm import tqdm
import os
import time
import warnings
from lib.revin import RevIN
from torch.utils.tensorboard import SummaryWriter

warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        # norm
        self.revin = RevIN(self.args.enc_in, eps=1e-5, affine=False)
        model = dlinear.Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.loss == 'mse':
            criterition = nn.MSELoss()
        elif self.args.loss == 'mae':
            criterion = nn.L1Loss()
        else:
            criterion = nn.L1Loss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                if self.args.revin:
                    batch_x = self.revin(batch_x, mode='norm')

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                if self.args.revin:
                    outputs = self.revin(outputs, mode='denorm')

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:,
                                  f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        writer = SummaryWriter(log_dir='./runs/' + setting)

        train_data, train_loader = self._get_data(flag='train')
        if not self.args.train_only:
            vali_data, vali_loader = self._get_data(flag='val')
            test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in tqdm(range(self.args.train_epochs), desc="Training", unit="epoch"):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            print(f"\nEpoch {epoch + 1}, Start:")
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if self.args.revin:
                    batch_x = self.revin(batch_x, mode='norm')

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)

                else:
                    outputs = self.model(batch_x)
                if self.args.revin:
                    outputs = self.revin(outputs, mode='denorm')

                    # print(outputs.shape,batch_y.shape)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:,
                                  f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                # writer.add_scalar('loss/train/batch', loss, iter_count + epoch * len(train_loader))

                if (i + 1) % 200 == 0:
                    dis_str = "\titers: {0:>5d}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item())
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    dis_str += "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(speed, left_time)
                    print(dis_str)
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {}, cost time: {}".format(
                epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            writer.add_scalar('loss/train/epoch', train_loss, epoch)
            if not self.args.train_only:
                vali_loss = self.vali(vali_data, vali_loader, criterion)
                test_loss = self.vali(test_data, test_loader, criterion)
                writer.add_scalar('loss/vali/epoch', vali_loss, epoch)
                writer.add_scalar('loss/test/epoch', test_loss, epoch)

                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
                early_stopping(vali_loss, self.model, path)
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f}".format(
                    epoch + 1, train_steps, train_loss))
                early_stopping(train_loss, self.model, path)

            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)
        writer.close()
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(
                './checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        inputx = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                if self.args.revin:
                    batch_x = self.revin(batch_x, mode='norm')

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    outputs = self.model(batch_x)

                if self.args.revin:
                    outputs = self.revin(outputs, mode='denorm')

                f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:,
                                  f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                inputx.append(batch_x.detach().cpu().numpy())
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate(
                        (input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate(
                        (input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        if self.args.test_flop:
            test_params_flop((batch_x.shape[1], batch_x.shape[2]))
            exit()

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        inputx = np.concatenate(inputx, axis=0)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, rse, corr = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))
        f = open("result.txt", 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rse:{}, corr:{}'.format(mse, mae, rse, corr))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe,rse, corr]))
        np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)
        # np.save(folder_path + 'x.npy', inputx)
        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(tqdm(pred_loader, desc="Predicting", leave=True)):
                batch_x = batch_x.float().to(self.device)
                # batch_y = batch_y.float()
                if self.args.revin:
                    batch_x = self.revin(batch_x, mode='norm')
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(batch_x)
                else:
                    # batch_size, pred_len, f_dim
                    outputs = self.model(batch_x)
                if self.args.revin:
                    outputs = self.revin(outputs, mode='denorm')

                # f_dim = -1 if self.args.features == 'MS' else 0
                # print(outputs.shape,batch_y.shape)
                # b, pred_len, f_dim
                outputs = outputs[:, -self.args.pred_len:, :]
                # print("outputs", outputs.shape)

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        # batch_size*, pred_len, f_dim
        preds = np.concatenate(preds, axis=0)
        # batch_size*pred_len, f_dim
        preds = preds.reshape(preds.shape[0] * preds.shape[1], preds.shape[2])
        if (pred_data.scale):
            preds = pred_data.inverse_transform(preds)
        f_dim = -1 if self.args.features == 'MS' else 0
        preds = np.squeeze(preds[:, f_dim:])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        np.save(folder_path + 'real_prediction.npy', preds)

        date = "2024-06-26"
        # pred数据集构造
        target_dates = pd.date_range(
            start=date,
            periods=self.args.pred_len,
        )
        target_dates = [date.strftime("%Y-%m-%d") for date in target_dates]
        merged_df = self.data_organization(folder_path+'real_prediction.npy',
                                           target_dates=target_dates)
        df_res = self.cal_metrics(merged_df, groups=["psd>=0", "psd>=1", "psd<1"],
                                  target_dates=target_dates)
        df_res.to_parquet(folder_path + "metrics_res.parquet")
        print(df_res.groupby(['group_type']).mean(numeric_only=True))

        return

    def data_organization(self, pred_data_path, target_dates):
        group_ids = ['store_id', 'product_id']
        # load data
        dataset = load_dataset("Dingdong-Inc/FreshRetailNet-50K")
        df_train = dataset['train'].to_pandas()
        df_eval = dataset['eval'].to_pandas()
        np_pred = np.load(pred_data_path)
        df_train_psd = df_train.groupby(
            group_ids)['sale_amount'].mean().to_frame('psd')
        df_eval = pd.merge(df_eval, df_train_psd, on=group_ids)

        # col rename
        df_train = df_train.rename(columns={'dt': 'date'})
        df_eval = df_eval.rename(columns={'dt': 'date'})
        # sort（必须）
        df_train = df_train.sort_values(by=['store_id', 'product_id', 'date'])
        df_eval = df_eval.sort_values(by=['store_id', 'product_id', 'date'])

        unique_groups = df_train[['store_id', 'product_id']
                                 ].drop_duplicates().reset_index(drop=True)
        # 创建日期数据框
        date_df = pd.DataFrame({'date': target_dates})
        # 使用 merge 进行笛卡尔积
        expanded_df = unique_groups.assign(key=1).merge(
            date_df.assign(key=1), on='key').drop('key', axis=1)
        np_pred[np_pred < 0] = 0
        expanded_df['prediction'] = np_pred
        df_pred = expanded_df

        df_groudtruth = df_eval[['store_id', 'product_id',
                                 'date', 'sale_amount', 'stock_hour6_22_cnt', 'psd']]
        merged_df = pd.merge(df_groudtruth, df_pred, on=[
                             'store_id', 'product_id', 'date'])
        return merged_df

    def cal_metrics(self, df, target_dates, groups=["psd>=0"]):
        sample_cnt, preds_sum, actuals_sum, preds_mean, actuals_mean = {}, {}, {}, {}, {}
        acc, wape, wpe, mae, bias = {}, {}, {}, {}, {}
        res = pd.DataFrame()
        for group in groups:
            df1 = df.query(group)
            sub_res = pd.DataFrame(target_dates, columns=["date"])
            for target_date in target_dates:
                df2 = df1[(df1.date == target_date) &
                          (df1.stock_hour6_22_cnt == 0)]
                preds = df2.prediction
                actuals = df2.sale_amount
                # import pdb; pdb.set_trace()
                sample_cnt[target_date] = len(df2)
                # print("len_df2", len(df2))
                preds_sum[target_date] = preds.sum(axis=0)
                actuals_sum[target_date] = actuals.sum(axis=0)
                preds_mean[target_date] = preds.mean(axis=0)
                actuals_mean[target_date] = actuals.mean(axis=0)
                acc[target_date] = 1 - \
                    (preds - actuals).abs().sum(axis=0) / \
                    actuals.abs().sum(axis=0)
                wape[target_date] = (
                    preds - actuals).abs().sum(axis=0) / actuals.abs().sum(axis=0)
                wpe[target_date] = (
                    preds - actuals).sum(axis=0) / actuals.abs().sum(axis=0)
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
            sub_res.insert(0, 'group_type', group)
            # sub_res.insert(0, 'data_type', data_type)
            res = pd.concat([res, sub_res], ignore_index=True)
        return res
