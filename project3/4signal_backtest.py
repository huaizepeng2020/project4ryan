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
import os
os.environ['CASTLE_BACKEND'] ='pytorch'

from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import DAG, IIDSimulation
from castle.algorithms import DAG_GNN,GOLEM,Notears


def parse_args():
    parser = argparse.ArgumentParser(description="data_download_from_fred")

    # ===== download ===== #
    parser.add_argument('--used_T', type=int, default=6, help='The time length of used past datas, the unit is month')
    parser.add_argument('--retrain_T', type=int, default=3,
                        help='The time frequency the model retrain, the unit is month')

    parser.add_argument('--backtest_start', type=str, default="2023-01-01", help='The start time of backtest')
    parser.add_argument('--backtest_end', type=str, default="2024-05-01", help='The end time of backtest')
    return parser.parse_args()

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
    backtest_start = args.backtest_start
    backtest_end = args.backtest_end
    used_T = args.used_T
    retrain_T = args.retrain_T

    time_interval=3

    datas_return=pd.read_csv(f"datas/stocks_x_orginalreturn_{1}.csv",index_col=0)

    datas_new=pd.read_csv(f"datas/stocks_x_preprocess_{time_interval}.csv",index_col=0)
    data_index = list(datas_new.index)
    all_dates = datas_new[(datas_new.index >= backtest_start) & (datas_new.index <= backtest_end)].index

    num_stocks=len(datas_new.columns)

    pnl_all=[]
    pnllong_all=[]
    pnlshort_all=[]
    IC_all=[]
    time_all=[]
    alpha_all=[]
    for i in all_dates:
        time_all.append(i)
        end_date_idx_c = data_index.index(i)
        start_date_idx_c = end_date_idx_c - used_T

        with open(os.path.join("signals") + f'/{i}_{3}.pickle', 'rb') as file:
            alpha_signal=pickle.load(file)

        with open(os.path.join("graphs") + f'/{i}_{3}.pickle', 'rb') as file:
            alpha_signal1=pickle.load(file)

        alpha_all.append(alpha_signal)

        # position
        position_c=np.sign(alpha_signal-np.median(alpha_signal))/len(alpha_signal)

        # pnl
        groundtruth_return=datas_return.iloc[end_date_idx_c]
        pnl_once=position_c*groundtruth_return
        pnl_oneday=pnl_once.sum()

        pnl_onedaylong=pnl_once[position_c>0].sum()
        pnl_onedayshort=pnl_once[position_c<0].sum()

        IC_c = scipy.stats.pearsonr(alpha_signal, np.array(datas_return.iloc[end_date_idx_c]).reshape(-1))[0]

        pnl_all.append(pnl_oneday)
        pnllong_all.append(pnl_onedaylong)
        pnlshort_all.append(pnl_onedayshort)
        IC_all.append(IC_c)
        print(f"pnl on day {i} is {pnl_oneday}")

    pnl_all=np.array(pnl_all)
    pnllong_all=np.array(pnllong_all)
    pnlshort_all=np.array(pnlshort_all)
    pnl_mean=np.mean(pnl_all)
    SR=np.mean(pnl_all)/(np.std(pnl_all))*(252)**0.5

    IC_mean=np.mean(IC_all)

    print(f"SR:{SR} IC:{IC_mean}")

    # 画图
    plt.figure(dpi=400)
    plt.plot(time_all, np.cumsum(pnl_all), label=f"pnl per day :{str(pnl_mean*1e4)[:3]} bps")
    plt.legend()
    plt.grid(True)
    plt.xticks(time_all[::int(20)])
    plt.xticks(rotation=90)
    plt.title(f"SR:{SR} IC:{IC_mean}")
    plt.tight_layout()
    plt.show()

    # plt.figure(dpi=400)
    # plt.plot(time_all, np.cumsum(pnllong_all), label=f"pnl per day of long:{str(np.mean(pnllong_all)*1e4)[:3]} bps")
    # plt.legend()
    # plt.grid(True)
    # plt.xticks(time_all[::int(20)])
    # plt.xticks(rotation=90)
    # plt.title(f"Backtest of long position")
    # plt.tight_layout()
    # plt.show()
    #
    # plt.figure(dpi=400)
    # plt.plot(time_all, np.cumsum(pnlshort_all), label=f"pnl per day of short:{str(np.mean(pnlshort_all)*1e4)[:3]} bps")
    # plt.legend()
    # plt.grid(True)
    # plt.xticks(time_all[::int(20)])
    # plt.xticks(rotation=90)
    # plt.title(f"Backtest of short position")
    # plt.tight_layout()
    # plt.show()
