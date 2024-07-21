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

    parser.add_argument('--selection_method', type=str, default="[3,1,2]",
                        help='encode of selection method for fatures, 1 for IC test, 2 for correlation test, 3 for autocorrelation test')
    parser.add_argument('--selection_haperparameters', type=str, default="[[5],[0.1],[0.7]]",
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
    if method == 3:
        k = hyperparameters[0]

        # data_c2_diff=(data_c2[1:]-data_c2[:-1])/data_c2[:-1]
        data_c2_diff=data_c2
        data_c2_diff_sr=np.nanstd(data_c2_diff)/np.abs(np.nanmean(data_c2_diff))

        # acf_y = smt.stattools.acf(data_c2, nlags=1)[1]

        acf_x_all=[]
        for i in range(data_c1.shape[1]):
            data_c1_c = data_c1[:, i]
            data_c1_diff = data_c1_c
            # data_c1_diff = (data_c1_c[1:] - data_c1_c[:-1]) / data_c1_c[:-1]
            data_c1_diff_sr = np.nanstd(data_c1_diff) / np.abs(np.nanmean(data_c1_diff))

            # acf = smt.stattools.acf(data_c1[:, i], nlags=1)[1]
            acf_x_all.append(data_c1_diff_sr)
        acf_x_all=np.array(acf_x_all)

        idx_left=np.where((acf_x_all<=data_c2_diff_sr*k)==True)[0]
        output_c=acf_x_all

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
        symbol_c=i
        datas_new=pd.read_csv(f"datas/{i}_xy_clean.csv",index_col=0)

        # rolling_period=[3,5,10,20,40,60,80,100,120]
        rolling_period=list(range(5,80))

        datas_price=datas_new[[i,symbol_target]]

        factors_all={}
        for j in rolling_period:
            factor_name=f"cov_price_{j}"
            factors_all[factor_name]=cov_pd(np.array(datas_price[i]),np.array(datas_price[symbol_target]),j)

            factor_name=f"cov_return_{j}"
            factors_all[factor_name]=cov_pd(np.array(datas_price[i].pct_change(1).shift(1)),np.array(datas_price[symbol_target].pct_change(1).shift(1)),j)


            # factor_name=f"cov_std_{j}"
            # factors_all[factor_name]=np.array((datas_price[i].rolling(j).std()/datas_price[i].rolling(j).mean()))/np.array(datas_price[symbol_target].rolling(j).std()/datas_price[symbol_target].rolling(j).mean())
            #
            # factor_name=f"cov_skew_{j}"
            # factors_all[factor_name]=np.array(datas_price[i].rolling(j).skew())/np.array(datas_price[symbol_target].rolling(j).skew())
            #
            # factor_name=f"cov_kurt_{j}"
            # factors_all[factor_name]=np.array(datas_price[i].rolling(j).kurt())/np.array(datas_price[symbol_target].rolling(j).kurt())


            # factor_name=f"cov_mean_{j}"
            # factors_all[factor_name]=cov_pd(np.array(datas_price[i].rolling(j).mean()),np.array(datas_price[symbol_target].rolling(j).mean()),j)

            # factor_name=f"cov_std_{j}"
            # factors_all[factor_name]=cov_pd(np.array(datas_price[i].rolling(j).std()),np.array(datas_price[symbol_target].rolling(j).std()),j)

            # factor_name=f"cov_skew_{j}"
            # factors_all[factor_name]=cov_pd(np.array(datas_price[i].rolling(j).skew()),np.array(datas_price[symbol_target].rolling(j).skew()),j)
            #
            # factor_name=f"cov_kurt_{j}"
            # factors_all[factor_name]=cov_pd(np.array(datas_price[i].rolling(j).kurt()),np.array(datas_price[symbol_target].rolling(j).kurt()),j)


        factors_all = pd.DataFrame(factors_all,index=datas_price.index)
        factors_all=factors_all.fillna(0)
        factors_all=factors_all.shift(1)

        # select factors
        indexall = factors_all.index
        index4select = indexall[(indexall < "2023-06-01")&(indexall>"2020-06-01")]
        data_x_c = factors_all.loc[index4select]
        data_y = datas_new.loc[index4select]["IC"]

        factor_names=list(data_x_c.columns)

        idx_left = []
        idx_left_c_c = np.array(list(range(data_x_c.shape[1])))
        for i, j in zip(selection_method, selection_para):
            idx_left_c, _ = selection_function(np.array(data_x_c)[:, idx_left_c_c], np.array(data_y), i, j)
            if i == 1:
                IC_singlef = np.zeros(data_x_c.shape[1])
                IC_singlef[idx_left_c_c] = _
            if i == 2:
                IC_cor = np.zeros(shape=(data_x_c.shape[1], data_x_c.shape[1]))
                for ii, iii in enumerate(idx_left_c_c):
                    for jj, jjj in enumerate(idx_left_c_c):
                        IC_cor[iii, jjj] = _[ii, jj]
                IC_cor += np.eye(len(IC_cor))
            idx_left_c_c = idx_left_c_c[idx_left_c]
            idx_left.append(idx_left_c_c)
        idx_left_all = idx_left_c_c
        print(f"---After selction, we have {len(idx_left_all)} raw features for stock {symbol_c}.---")


        table_x = PrettyTable()
        table_x.title = f'Selected factors of {symbol_c}'
        table_x.field_names = ["ID", "Name", "its IC with target"]

        name_all = []
        for idx, ii in enumerate(idx_left_all):
            name_c = data_x_c.columns[ii]
            name_all.append(name_c)
            IC_c = IC_singlef[ii]

            table_x.add_row([idx, name_c, IC_c])
        print(table_x)

        # print("---------Their correlation matrix is as---------")
        # print(IC_cor[idx_left_all][:,idx_left_all])
        # print("------------------")
        ICs = IC_cor[idx_left_all][:, idx_left_all]
        # 设置热力图的颜色范围
        cmap = sns.color_palette("Blues")
        # 创建热力图
        f, ax = plt.subplots(figsize=(6, 6))
        heatmap = sns.heatmap(ICs, cmap=cmap, vmax=1.0, square=True, annot=True,ax=ax)
        # 设置轴标签
        ax.set_xticks(np.arange(ICs.shape[1]) + 0.5, minor=False)
        ax.set_yticks(np.arange(ICs.shape[0]) + 0.5, minor=False)
        ax.invert_yaxis()
        # plt.title(str(dict(zip(range(len(name_all)),name_all))))
        cbar = ax.collections[0].colorbar
        cbar.set_label('Intensity')
        plt.show()

        data_x_left = factors_all[factors_all.columns[idx_left_all]]

        data_x_left = pd.concat([data_x_left,datas_new["IC"]],axis=1)

        data_x_left.to_csv(f"datas/{symbol_c}_selectedfactors_y.csv")
