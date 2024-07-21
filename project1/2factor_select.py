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
import statsmodels.tsa.api as smt
from statsmodels.tsa.stattools import adfuller

# time.sleep(10)
def parse_args():
    parser = argparse.ArgumentParser(description="data_download_from_fred")

    # ===== download ===== #
    parser.add_argument('--my_api', type=str, default="34e95c57aae8485f53fef1a429cef662", help='API access of FRED')
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

        data_c2_diff=(data_c2[1:]-data_c2[:-1])/data_c2[:-1]
        data_c2_diff_sr=np.nanstd(data_c2_diff)/np.nanmean(data_c2_diff)

        # acf_y = smt.stattools.acf(data_c2, nlags=1)[1]

        acf_x_all=[]
        for i in range(data_c1.shape[1]):
            data_c1_c = data_c1[:, i]
            data_c1_diff = (data_c1_c[1:] - data_c1_c[:-1]) / data_c1_c[:-1]
            data_c1_diff_sr = np.nanstd(data_c1_diff) / np.nanmean(data_c1_diff)

            # acf = smt.stattools.acf(data_c1[:, i], nlags=1)[1]
            acf_x_all.append(data_c1_diff_sr)
        acf_x_all=np.array(acf_x_all)

        idx_left=np.where((acf_x_all<=data_c2_diff_sr*k)==True)[0]
        output_c=acf_x_all

    return idx_left,output_c

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
    selection_method = args.selection_method
    selection_method = eval(selection_method)
    selection_para = args.selection_haperparameters
    selection_para = eval(selection_para)
    my_api = args.my_api

    # load data
    data_x=pd.read_csv("datas/factors_clean.csv",index_col=0)
    data_y=pd.read_csv("datas/target_clean.csv",index_col=0)

    # to avoid look-ahead information
    data_x=data_x.shift(1)


    # use part his to avoid introducing out-of-sample information
    indexall=data_y.index
    index4select=indexall[indexall<"2000-01-01"]
    data_x_c=data_x.loc[index4select]
    data_y=data_y.loc[index4select]

    idx_left=[]
    idx_left_c_c=np.array(list(range(data_x_c.shape[1])))
    for i,j in zip(selection_method,selection_para):
        # idx_left_c,_=selection_function(np.array(data_x_c),np.array(data_y),i,j)
        # idx_left.append(idx_left_c)
        idx_left_c,_=selection_function(np.array(data_x_c)[:,idx_left_c_c],np.array(data_y),i,j)
        if i ==1:
            IC_singlef=np.zeros(data_x_c.shape[1])
            IC_singlef[idx_left_c_c]=_
            # IC_singlef=_
        if i ==2:
            IC_cor=np.zeros(shape=(data_x_c.shape[1],data_x_c.shape[1]))
            for ii,iii in enumerate(idx_left_c_c):
                for jj, jjj in enumerate(idx_left_c_c):
                    IC_cor[iii,jjj]=_[ii,jj]
            IC_cor+=np.eye(len(IC_cor))
            # IC_cor=_
        idx_left_c_c=idx_left_c_c[idx_left_c]
        idx_left.append(idx_left_c_c)

    # idx_left_all=list(idx_left[0])
    # for i in idx_left:
    #     idx_left_all = list(set(idx_left_all)&set(i))
    idx_left_all=idx_left_c_c
    print(f"---After selction, we have {len(idx_left_all)} raw features.---")

    table_x = PrettyTable()
    table_x.title = 'Selected factors'
    table_x.field_names = ["ID","Fred ID", "Full name", "its IC with target"]

    name_all=[]
    for idx,i in enumerate(idx_left_all):
        name_c=data_x_c.columns[i]
        fred = Fred(api_key=my_api)
        infor_i = fred.get_series_info(name_c)
        title_c=infor_i["title"]
        name_all.append(name_c)
        IC_c=IC_singlef[i]

        table_x.add_row([idx,name_c, title_c, IC_c])
    print(table_x)

    # print("---------Their correlation matrix is as---------")
    # print(IC_cor[idx_left_all][:,idx_left_all])
    # print("------------------")
    ICs=IC_cor[idx_left_all][:,idx_left_all]
    # 设置热力图的颜色范围
    cmap = sns.color_palette("Blues")
    # 创建热力图
    f, ax = plt.subplots(figsize=(6, 6))
    heatmap = sns.heatmap(ICs, cmap=cmap, vmax=1.0, square=True,annot=True, ax=ax)
    # 设置轴标签
    ax.set_xticks(np.arange(ICs.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(ICs.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()
    # plt.title(str(dict(zip(range(len(name_all)),name_all))))
    cbar = ax.collections[0].colorbar
    cbar.set_label('Intensity')
    plt.title("IC between selected factors")
    plt.show()

    data_x_left = data_x[data_x.columns[idx_left_all]]

    data_x_left.to_csv("datas/factors_select.csv")