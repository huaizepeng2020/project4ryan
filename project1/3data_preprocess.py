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

def parse_args():
    parser = argparse.ArgumentParser(description="data_download_from_fred")

    # ===== download ===== #
    parser.add_argument('--combine_method', type=str, default="[1,2,3,4]",
                        help='encode of combing method for features, 1 for LR regression, 2 for lightgbm, 3 for MLP based Neural Network,4 for FM based Neural Network')
    parser.add_argument('--used_T', type=int, default=1, help='The time length of used past datas, the unit is month')
    parser.add_argument('--train_T', type=int, default=9, help='The time length of training datas, the unit is month')
    parser.add_argument('--valid_T', type=int, default=3, help='The time length of validation datas, the unit is month')
    parser.add_argument('--retrain_T', type=int, default=1, help='The time frequency the model retrain, the unit is month')
    parser.add_argument('--backtest_start', type=str, default="1990-01-01", help='The start time of backtest')
    parser.add_argument('--backtest_end', type=str, default="2024-07-01", help='The end time of backtest')
    return parser.parse_args()

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
    combine_method = args.combine_method
    combine_method = eval(combine_method)
    used_T = args.used_T
    train_T = args.train_T
    valid_T = args.valid_T
    retrain_T = args.retrain_T
    backtest_start = args.backtest_start
    backtest_end = args.backtest_end

    # load data
    data_x=pd.read_csv("datas/factors_select.csv",index_col=0)
    data_y=pd.read_csv("datas/target_clean.csv",index_col=0)

    # de-mean operation for time series datas
    data_x = data_x/data_x.loc["2000-01-01"]


    # cut-off for some extremums and outliers
    flag_cutoff=True
    while flag_cutoff:
        kk = 1
        data_x_s1=data_x.shift(1)
        data_x_diff1 = data_x.diff(1).abs().rolling(12).mean().shift(1)
        data_x_diff2 = data_x.diff(1).abs().rolling(12).std().shift(1)
        data_x_falg1 = data_x > data_x_s1 + data_x_diff1 + kk*data_x_diff1
        data_x_falg2 = data_x < data_x_s1 - data_x_diff1 - kk*data_x_diff1
        data_x_falg1[data_x_falg1.index<"1998-01-01"]=False
        data_x_falg2[data_x_falg2.index<"1998-01-01"]=False
        data_x[data_x_falg1] = (data_x_s1 + data_x_diff1 + kk*data_x_diff1)[data_x_falg1]
        data_x[data_x_falg2] = (data_x_s1 - data_x_diff1 - kk*data_x_diff1)[data_x_falg2]

        data_x_s1=data_x.shift(1)
        data_x_diff1 = data_x.diff(1).abs().rolling(12).mean().shift(1)
        data_x_diff2 = data_x.diff(1).abs().rolling(12).std().shift(1)
        data_x_falg1 = data_x > data_x_s1 + data_x_diff1 + kk*data_x_diff1
        data_x_falg2 = data_x < data_x_s1 - data_x_diff1 - kk*data_x_diff1
        data_x_falg1[data_x_falg1.index<"1998-01-01"]=False
        data_x_falg2[data_x_falg2.index<"1998-01-01"]=False

        if np.sum(np.sum(data_x_falg1)+np.sum(data_x_falg2))==0:
            flag_cutoff=False
    # data_x = data_x.rolling(6).mean()

    # cut-off for some extremums and outliers
    # kk=1
    # data_x_mean=data_x.rolling(24).quantile(0.5).shift(1)
    # data_x_std=data_x.diff(1).abs().rolling(24).quantile(0.5).shift(1)*2
    # data_x_falg1=data_x>data_x_mean+kk*data_x_std
    # data_x_falg2=data_x<data_x_mean-kk*data_x_std
    # data_x[data_x_falg1]=(data_x_mean+kk*data_x_std)[data_x_falg1]
    # data_x[data_x_falg2]=(data_x_mean-kk*data_x_std)[data_x_falg2]



    # data_x_diff=data_x.diff(1)
    # data_x_diff=data_x_diff.fillna(0)
    # data_x_past_std=data_x_diff.rolling(12).std()
    # data_x_past_std=data_x_past_std.shift(1)
    # data_x_diff_flag=data_x_diff.abs()>3*data_x_past_std
    # data_x_diff[data_x_diff_flag]=3*data_x_past_std[data_x_diff_flag]*np.sign(data_x_diff)
    # data_x[data_x_diff_flag]
    # data_x = data_x.rolling(12).mean()

    data_x.to_csv("datas/factors_preprocess.csv")