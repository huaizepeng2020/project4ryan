import numpy as np
import pandas as pd
import random
import os
import scipy.stats
import torch
import time
from prettytable import PrettyTable
from tqdm import tqdm
import pickle
import gc
import argparse
from fredapi import Fred
import datetime
from datetime import timedelta
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import torch.nn as nn
from collections import OrderedDict
from base_model import *
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="data_download_from_fred")

    # ===== download ===== #
    parser.add_argument('--symbols', type=str, default="[ 'GOOG', 'AMZN', 'JPM', 'GME', 'XOM']", help='concerned symbols')
    # parser.add_argument('--symbols', type=str, default="[ 'GME', 'XOM']", help='concerned symbols')
    parser.add_argument('--symbol_target', type=str, default="SPY", help='target symbol')
    parser.add_argument('--period_IC', type=int, default=40, help='IC during future two months')

    parser.add_argument('--combine_method', type=str, default="[1,2,3,4,5,6]",
                        help='encode of combing method for features, 1 for LR regression, 2 for lightgbm, 3 for MLP based Neural Network,4 for FM based Neural Network, 5 for GRU based Neural Network, 6  for Transformer')
    parser.add_argument('--used_T', type=int, default=5, help='The time length of used past datas, the unit is month')
    parser.add_argument('--train_T', type=int, default=120, help='The time length of training datas, the unit is month')
    parser.add_argument('--valid_T', type=int, default=40, help='The time length of validation datas, the unit is month')
    parser.add_argument('--retrain_T', type=int, default=1, help='The time frequency the model retrain, the unit is month')
    # parser.add_argument('--backtest_start', type=str, default="1990-01-01", help='The start time of backtest')
    parser.add_argument('--backtest_start', type=str, default="2024-01-01", help='The start time of backtest')
    parser.add_argument('--backtest_end', type=str, default="2024-05-01", help='The end time of backtest')
    return parser.parse_args()


def rolling_window(a, window):
  shape = (a.shape[0]-window+1, window, a.shape[1])
  strides = (a.strides[0],) + a.strides
  return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

if __name__ == "__main__":
    path_c_abs=os.getcwd()
    """fix the random seed"""
    seed = 2024
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    global args
    args = parse_args()
    symbols=eval(args.symbols)
    symbol_target=args.symbol_target
    period_IC=args.period_IC

    combine_method = args.combine_method
    combine_method = eval(combine_method)
    used_T = args.used_T
    train_T = args.train_T
    valid_T = args.valid_T
    retrain_T = args.retrain_T
    backtest_start = args.backtest_start
    backtest_end = args.backtest_end

    for ii in symbols:
        # datas_new=pd.read_csv(f"datas/{ii}_preprocessedfactors_y.csv",index_col=0)
        datas_new=pd.read_csv(f"datas/{ii}_selectedfactors_y.csv",index_col=0)

        data_x=datas_new.drop("IC",axis=1)
        data_y=datas_new["IC"]

        # data_x=np.log(data_x+1)
        # data_x[data_x<=-0.5]=-0.5
        # data_y=np.log(data_y+1)
        # data_y[data_y <= -0.5] = -0.5
        # data_y=data_y.shift(-10).rolling(10).mean()

        all_dates=data_y[(data_y.index>=backtest_start)&(data_y.index<=backtest_end)].index

        idx_c = 0

        data_x_index = list(data_x.index)
        data_y_index = list(data_y.index)

        factor_name = list(data_x.columns)

        for i in all_dates[::retrain_T]:
            end_date_idx_y = data_y_index.index(i)-period_IC
            end_date_idx_yc = data_y_index.index(i)
            start_date_idx1_y = end_date_idx_y - valid_T
            start_date_idx0_y = end_date_idx_y - train_T - valid_T-period_IC

            end_date_idx = data_x_index.index(i)-period_IC
            end_date_idxc = data_x_index.index(i)
            start_date_idx1 = end_date_idx - valid_T
            start_date_idx0 = end_date_idx - train_T - valid_T-period_IC

            train_datas_x = data_x.iloc[start_date_idx0:start_date_idx1-period_IC]
            valid_datas_x = data_x.iloc[start_date_idx1:end_date_idx]
            train_datas_y = data_y.iloc[start_date_idx0_y:start_date_idx1_y-period_IC]
            valid_datas_y = data_y.iloc[start_date_idx1_y:end_date_idx_y]
            factor_data = data_x.iloc[end_date_idxc]
            target_data = data_y.iloc[end_date_idx_yc]

            train_datas_x = np.array(train_datas_x)
            valid_datas_x = np.array(valid_datas_x)
            train_datas_y = np.array(train_datas_y)
            valid_datas_y = np.array(valid_datas_y)
            factor_data = np.array(factor_data)[np.newaxis, :]
            target_data = np.array(target_data)

            train_datas_x_time = rolling_window(np.array(data_x.iloc[start_date_idx0 - used_T + 1:start_date_idx1-period_IC]),
                                                used_T)
            valid_datas_x_time = rolling_window(np.array(data_x.iloc[start_date_idx1 - used_T + 1:end_date_idx]),
                                                used_T)
            factor_data_time = np.array(data_x[end_date_idxc - used_T + 1:end_date_idxc + 1])

            # train model and predict
            for j in combine_method:
                model_c = models_mh(j, factor_name)
                if not os.path.exists(os.path.join("models", model_c.model.name)):
                    os.makedirs(os.path.join("models", model_c.model.name))
                if not os.path.exists(os.path.join("signals", model_c.model.name)):
                    os.makedirs(os.path.join("signals", model_c.model.name))

                if model_c.model.time_falg:
                    model_c.train(train_datas_x_time, train_datas_y, valid_datas_x_time, valid_datas_y)
                else:
                    model_c.train(train_datas_x, train_datas_y, valid_datas_x, valid_datas_y)

                with open(os.path.join("models", model_c.model.name) + f'/{ii}_{i}.pickle', 'wb') as file:
                    pickle.dump(model_c, file)
                if model_c.model.time_falg:
                    # bias_all = []
                    # for idx_c in range(len(valid_datas_x_time)):
                    #     bias_c = model_c.predict(valid_datas_x_time[idx_c])-valid_datas_y[idx_c]
                    #     bias_all.append(bias_c)
                    # bias_all = np.nanmean(bias_all)
                    # predict_data_c = model_c.predict(factor_data_time)-bias_all
                    predict_data_c = model_c.predict(factor_data_time)
                else:
                    # bias_all=[]
                    # for idx_c in range(len(valid_datas_x)):
                    #     bias_c = model_c.predict(valid_datas_x[idx_c][np.newaxis,:])-valid_datas_y[idx_c]
                    #     bias_all.append(bias_c)
                    # bias_all=np.nanmean(bias_all)
                    # predict_data_c = model_c.predict(factor_data)-bias_all
                    predict_data_c = model_c.predict(factor_data)
                # predict_data_c=np.exp(predict_data_c)-1
                with open(os.path.join("signals", model_c.model.name) + f'/{ii}_{i}.pickle', 'wb') as file:
                    pickle.dump(predict_data_c, file)

            print(ii,i, " finished")

    a=1