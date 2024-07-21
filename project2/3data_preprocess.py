import numpy as np
import pandas as pd
import random

import scipy.stats
import torch
import time
from prettytable import PrettyTable
from tqdm import tqdm
import pickle
import gc
import argparse
from fredapi import Fred
import matplotlib.pyplot as plt
import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser(description="data_download_from_fred")

    # ===== download ===== #
    parser.add_argument('--symbols', type=str, default="[ 'GOOG', 'AMZN', 'JPM', 'GME', 'XOM']", help='concerned symbols')
    parser.add_argument('--symbol_target', type=str, default="SPY", help='target symbol')
    parser.add_argument('--period_IC', type=int, default=40, help='IC during future two months')

    parser.add_argument('--clean_way_4x', type=str, default="[1,3]",
                        help='encode of clean method for fatures, 1 for forward filling, 2 for back filling, 3 for zero filling')

    parser.add_argument('--selection_method', type=str, default="[1,2]",
                        help='encode of selection method for fatures, 1 for IC test, 2 for correlation test')
    parser.add_argument('--selection_haperparameters', type=str, default="[[0.1],[0.7]]",
                        help='encode of clean method for targets, 1 for forward filling, 2 for back filling, 3 for zero filling, 4 for interpolation ')
    return parser.parse_args()


def selection_function(data_c1,data_c2,method,hyperparameters):
    if method == 1:
        IC_min=hyperparameters[0]
        IC_all=[]
        for i in range(data_c1.shape[1]):
            IC_all.append(scipy.stats.pearsonr(data_c1[:,i].reshape(-1),data_c2.reshape(-1))[0])
        IC_all=np.array(IC_all)
        IC_all=np.nan_to_num(IC_all)

        output_c=IC_all
        idx_left = np.where(np.abs(IC_all)>=IC_min)[0]
    if method == 2:
        IC_min = hyperparameters[0]

        IC_all=[]
        for i in range(data_c1.shape[1]):
            IC_all.append(scipy.stats.pearsonr(data_c1[:,i].reshape(-1),data_c2.reshape(-1))[0])
        IC_all=np.array(IC_all)

        ICs=np.corrcoef(data_c1.T)
        ICs -= np.eye(len(ICs))
        ICs=np.nan_to_num(ICs,nan=1, posinf=1, neginf=1)
        ICs=np.abs(ICs)

        output_c=ICs
        # ICs_one=np.sum(ICs,axis=1)
        idx_high=np.where(ICs>=IC_min)

        idx_sort=ICs[idx_high]
        idx_sort=np.argsort(idx_sort)[::-1]

        idx_del=[]
        for idx_c in idx_sort.tolist():
            i,j=idx_high[0][idx_c],idx_high[1][idx_c]
            if j not in idx_del and i not in idx_del:
                if IC_all[i]>=IC_all[j]:
                    idx_del.append(j)
                else:
                    idx_del.append(i)
        idx_del=list(set(idx_del))
        idx_left=[i for i in range(len(ICs)) if i not in idx_del]

    return idx_left,output_c


def cov_pd(data1,data2,rolling_w):
    data1_c = np.lib.stride_tricks.sliding_window_view(data1, rolling_w)
    data2_c = np.lib.stride_tricks.sliding_window_view(data2, rolling_w)

    data1_c_error=data1_c-np.mean(data1_c,axis=1,keepdims=True)
    data2_c_error=data2_c-np.mean(data2_c,axis=1,keepdims=True)

    data1_c_std=np.std(data1_c,axis=1,keepdims=True)
    data2_c_std=np.std(data2_c,axis=1,keepdims=True)

    IC=np.mean((data1_c_error*data2_c_error)/(data1_c_std*data2_c_std),axis=1)

    IC=np.concatenate([np.zeros(rolling_w-1),IC])

    return IC

if __name__ == "__main__":
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
    selection_method = args.selection_method
    selection_method = eval(selection_method)
    selection_para = args.selection_haperparameters
    selection_para = eval(selection_para)

    for i in symbols:
        data_x_left=pd.read_csv(f"datas/{i}_selectedfactors_y.csv",index_col=0)

        # data_x_left["IC"]=data_x_left["IC"].shift(-10).rolling(10).mean()
        # data_x_left=data_x_left.fillna(0)
        # data_x=np.log(data_x_left+1)
        # data_x[data_x<=-0.5]=-0.5

        # data_y=np.log(data_y+1)
        # data_y[data_y <= -0.5] = -0.5
        # data_y=data_y.shift(-10).rolling(10).mean()

        # data_x=data_x_left.rolling(10).mean()
        # data_x=data_x_left.rolling(1).mean()
        # data_x["IC"]=data_x_left["IC"]
        # data_x1=data_x[(data_x.index < "2022-06-01")&(data_x.index>"2020-06-01")].mean(axis=0).abs()
        # data_x1=data_x1/data_x1["IC"]
        # data_x=data_x/data_x1

        data_x=data_x_left

        # rolling operation to smooth factors
        # data_x = data_x / data_x.loc["2020-06-01"]

        data_x.to_csv(f"datas/{i}_preprocessedfactors_y.csv")
