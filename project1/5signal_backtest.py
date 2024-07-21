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
from tabulate import tabulate
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description="data_download_from_fred")

    # ===== download ===== #
    parser.add_argument('--combine_method', type=str, default="[1,2,3,4,5,6]",
                        help='encode of combing method for features, 1 for LR regression, 2 for lightgbm, 3 for MLP based Neural Network,4 for FM based Neural Network')
    parser.add_argument('--combine_method_name', type=str, default="['OLS','Lightgbm','MLP','FM','GRU','Transformer']", help='The name of each model')
    parser.add_argument('--used_T', type=int, default=1, help='The time length of used past datas, the unit is month')
    parser.add_argument('--train_T', type=int, default=9, help='The time length of training datas, the unit is month')
    parser.add_argument('--valid_T', type=int, default=3, help='The time length of validation datas, the unit is month')
    parser.add_argument('--retrain_T', type=int, default=1, help='The time frequency the model retrain, the unit is month')
    parser.add_argument('--backtest_start', type=str, default="2000-01-01", help='The start time of backtest')
    parser.add_argument('--backtest_end', type=str, default="2024-07-01", help='The end time of backtest')
    # parser.add_argument('--backtest_end', type=str, default="2019-11-01", help='The end time of backtest')
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
    combine_method_name = args.combine_method_name
    combine_method_name = eval(combine_method_name)
    used_T = args.used_T
    train_T = args.train_T
    valid_T = args.valid_T
    retrain_T = args.retrain_T
    backtest_start = args.backtest_start
    backtest_end = args.backtest_end

    combine_method_dict=dict(zip(combine_method,combine_method_name))

    # load data
    data_x=pd.read_csv("datas/factors_preprocess.csv",index_col=0)
    data_y=pd.read_csv("datas/target_clean.csv",index_col=0)

    all_dates=data_y[(data_y.index>=backtest_start)&(data_y.index<=backtest_end)].index

    all_predict_datas=[]
    for i in all_dates[::retrain_T]:
        all_predict_datas_c = []
        all_predict_datas_c.append(data_y.loc[i])
        for j in combine_method:
            model_name=combine_method_dict[j]
            with open(os.path.join("signals",model_name)+f'/{i}.pickle', 'rb') as file:
                predict_data_c=pickle.load(file)
            all_predict_datas_c.append(predict_data_c)
        # all_predict_datas_c=np.array(all_predict_datas_c).reshape(-1)
        all_predict_datas.append(all_predict_datas_c)

    # all_predict_datas=np.array(all_predict_datas)
    all_predict_datas=np.array(all_predict_datas).squeeze(-1)

    signals_pd = pd.DataFrame(columns=["grooundtruth"] + combine_method_name, data=all_predict_datas)
    ema_values = []
    for window_size in range(1, len(signals_pd) + 1):
        ema = signals_pd[:window_size].ewm(alpha=1 / 3, adjust=False).mean().iloc[-1]
        ema_values.append(ema)
    signals_pd1 = pd.DataFrame(ema_values)[combine_method_name[2:]]
    signals_pd[combine_method_name[2:]] = signals_pd1

    # plot prediction and calculate metric
    metrics_name = ["IC","MAE","RMSE"]
    # metrics_name = ["IC","MAE","std_of_error","std_of_signal"]
    metrics_dict = pd.DataFrame(index=combine_method_name,columns=metrics_name)
    for model_c in combine_method_name:
        for metric_c in metrics_name:
            if metric_c=="IC":
                metric0=scipy.stats.pearsonr(signals_pd["grooundtruth"],signals_pd[model_c])[0]
            elif metric_c=="MAE":
                metric0=np.mean(np.abs(signals_pd["grooundtruth"]-signals_pd[model_c]))
            elif metric_c=="std_of_error":
                metric0=np.std(np.abs(signals_pd["grooundtruth"]-signals_pd[model_c]))
            elif metric_c=="RMSE":
                metric0=np.mean((signals_pd[model_c]-signals_pd["grooundtruth"])**2)**0.5
            metrics_dict.loc[model_c][metric_c]=metric0

    print(metrics_dict.to_string())

    # 画图
    plt.figure(dpi=400)
    plt.plot(all_dates, signals_pd["grooundtruth"], label="groundtruth")
    for model_c in combine_method_name:
        plt.plot(all_dates, signals_pd[model_c], label=model_c,linewidth=1)
    plt.legend()
    plt.grid(True)
    plt.xticks(all_dates[::12])
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()


    for model_c in combine_method_name:
        plt.figure(dpi=400)
        plt.plot(all_dates, signals_pd["grooundtruth"], label="groundtruth")
        plt.plot(all_dates, signals_pd[model_c], label=model_c,linewidth=1)
        plt.legend()
        plt.grid(True)
        plt.xticks(all_dates[::12])
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()

    a=1