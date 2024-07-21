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
    parser.add_argument('--symbols', type=str, default="[ 'GOOG', 'AMZN', 'JPM', 'GME', 'XOM']", help='concerned symbols')
    parser.add_argument('--symbol_target', type=str, default="SPY", help='target symbol')
    parser.add_argument('--period_IC', type=int, default=40, help='IC during future two months')

    # parser.add_argument('--combine_method', type=str, default="[2,3,4,5,6]",
    parser.add_argument('--combine_method', type=str, default="[1,2,3,4,5,6]",
                        help='encode of combing method for features, 1 for LR regression, 2 for lightgbm, 3 for MLP based Neural Network,4 for FM based Neural Network')
    # parser.add_argument('--combine_method_name', type=str, default="['FM','GRU','Transformer']", help='The name of each model')
    # parser.add_argument('--combine_method_name', type=str, default="['OLS','Lightgbm','MLP','GRU','Transformer']", help='The name of each model')
    parser.add_argument('--combine_method_name', type=str, default="['OLS','Lightgbm','MLP','FM','GRU','Transformer']", help='The name of each model')
    parser.add_argument('--used_T', type=int, default=1, help='The time length of used past datas, the unit is month')
    parser.add_argument('--train_T', type=int, default=9, help='The time length of training datas, the unit is month')
    parser.add_argument('--valid_T', type=int, default=3, help='The time length of validation datas, the unit is month')
    parser.add_argument('--retrain_T', type=int, default=1, help='The time frequency the model retrain, the unit is month')
    # parser.add_argument('--backtest_start', type=str, default="2023-01-01", help='The start time of backtest')
    parser.add_argument('--backtest_start', type=str, default="2024-01-01", help='The start time of backtest')
    parser.add_argument('--backtest_end', type=str, default="2024-04-01", help='The end time of backtest')
    return parser.parse_args()

def test_metric(combine_method_name_c):
    for model_c in combine_method_name_c:
        for metric_c in metrics_name:
            if metric_c == "IC":
                metric0 = scipy.stats.pearsonr(signals_pd["grooundtruth"], signals_pd[model_c])[0]
            elif metric_c == "MAE":
                metric0 = np.mean(np.abs(signals_pd["grooundtruth"] - signals_pd[model_c]))
            elif metric_c == "MAE10":
                metric0 = np.mean(np.abs(signals_pd["grooundtruth_mean10"] - signals_pd[model_c]))
            elif metric_c == "MAE20":
                metric0 = np.mean(np.abs(signals_pd["grooundtruth_mean20"] - signals_pd[model_c]))
            elif metric_c == "correct_rate":
                metric0 = np.sum(np.abs(signals_pd["grooundtruth"] - signals_pd[model_c]) <= 0.3) / len(signals_pd)
            elif metric_c == "quantile_error_0.2/4/6/8":
                metric01 = np.quantile(np.abs(signals_pd["grooundtruth"] - signals_pd[model_c]), 0.2)
                metric02 = np.quantile(np.abs(signals_pd["grooundtruth"] - signals_pd[model_c]), 0.4)
                metric03 = np.quantile(np.abs(signals_pd["grooundtruth"] - signals_pd[model_c]), 0.6)
                metric04 = np.quantile(np.abs(signals_pd["grooundtruth"] - signals_pd[model_c]), 0.8)
                metric0 = [metric01, metric02, metric03, metric04]
            # elif metric_c=="std_of_error":
            #     metric0=np.std(np.abs(signals_pd["grooundtruth"]-signals_pd[model_c]))
            # elif metric_c=="std_of_signal":
            #     metric0=np.std(signals_pd[model_c])/np.std(signals_pd["grooundtruth"])
            metrics_dict.loc[model_c][metric_c] = metric0

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
    combine_method_name = args.combine_method_name
    combine_method_name = eval(combine_method_name)
    used_T = args.used_T
    train_T = args.train_T
    valid_T = args.valid_T
    retrain_T = args.retrain_T
    backtest_start = args.backtest_start
    backtest_end = args.backtest_end

    combine_method_dict=dict(zip(combine_method,combine_method_name))

    for ii in symbols:
        datas_new=pd.read_csv(f"datas/{ii}_selectedfactors_y.csv",index_col=0)

        # data_x=datas_new.drop("IC",axis=1)
        data_y=datas_new["IC"]
        # data_y = data_y.shift(-20).rolling(20).mean()

        all_dates=data_y[(data_y.index>=backtest_start)&(data_y.index<=backtest_end)].index

        idx_c = 0

        # data_x_index = list(data_x.index)
        data_y_index = list(data_y.index)

        # factor_name = list(data_x.columns)

        all_predict_datas = []
        for i in all_dates[::retrain_T]:
            all_predict_datas_c = []
            all_predict_datas_c.append(data_y.loc[i])
            for j in combine_method:
                model_name=combine_method_dict[j]
                with open(os.path.join("signals",model_name)+f'/{ii}_{i}.pickle', 'rb') as file:
                    predict_data_c=pickle.load(file).reshape(-1)[0]
                all_predict_datas_c.append(predict_data_c)
            all_predict_datas.append(all_predict_datas_c)

        all_predict_datas=np.array(all_predict_datas)
        # all_predict_datas=np.array(all_predict_datas).squeeze(-1)

        signals_pd=pd.DataFrame(columns=["grooundtruth"]+combine_method_name,data=all_predict_datas)
        # ema_values = []
        # for window_size in range(1, len(signals_pd) + 1):
        #     ema = signals_pd[:window_size].ewm(alpha=1 / 5, adjust=False).mean().iloc[-1]
        #     ema_values.append(ema)

        signals_pd1=signals_pd.rolling(5).mean()
        signals_pd1=signals_pd1.fillna(0)
        signals_pd[combine_method_name]=signals_pd1[combine_method_name]
        signals_pd["grooundtruth_mean10"]=signals_pd["grooundtruth"].shift(-10).rolling(10).mean()
        signals_pd["grooundtruth_mean20"]=signals_pd["grooundtruth"].shift(-20).rolling(20).mean()

        # plot prediction and calculate metric
        metrics_name = ["IC","MAE","quantile_error_0.2/4/6/8"]
        # metrics_name = ["IC","MAE","std_of_error","std_of_signal"]
        metrics_dict = pd.DataFrame(index=combine_method_name+["assemble"],columns=metrics_name)
        test_metric(combine_method_name)
        # for model_c in combine_method_name:
        #     for metric_c in metrics_name:
        #         if metric_c=="IC":
        #             metric0=scipy.stats.pearsonr(signals_pd["grooundtruth"],signals_pd[model_c])[0]
        #         elif metric_c=="MAE":
        #             metric0=np.mean(np.abs(signals_pd["grooundtruth"]-signals_pd[model_c]))
        #         elif metric_c=="MAE10":
        #             metric0=np.mean(np.abs(signals_pd["grooundtruth_mean10"]-signals_pd[model_c]))
        #         elif metric_c=="MAE20":
        #             metric0=np.mean(np.abs(signals_pd["grooundtruth_mean20"]-signals_pd[model_c]))
        #         elif metric_c=="correct_rate":
        #             metric0=np.sum(np.abs(signals_pd["grooundtruth"]-signals_pd[model_c])<=0.3)/len(signals_pd)
        #         elif metric_c=="quantile_error_0.2/4/6/8":
        #             metric01=np.quantile(np.abs(signals_pd["grooundtruth"]-signals_pd[model_c]),0.2)
        #             metric02=np.quantile(np.abs(signals_pd["grooundtruth"]-signals_pd[model_c]),0.4)
        #             metric03=np.quantile(np.abs(signals_pd["grooundtruth"]-signals_pd[model_c]),0.6)
        #             metric04=np.quantile(np.abs(signals_pd["grooundtruth"]-signals_pd[model_c]),0.8)
        #             metric0=[metric01,metric02,metric03,metric04]
        #         # elif metric_c=="std_of_error":
        #         #     metric0=np.std(np.abs(signals_pd["grooundtruth"]-signals_pd[model_c]))
        #         # elif metric_c=="std_of_signal":
        #         #     metric0=np.std(signals_pd[model_c])/np.std(signals_pd["grooundtruth"])
        #         metrics_dict.loc[model_c][metric_c]=metric0

        # choose best model
        best1_model=metrics_dict["IC"][metrics_dict["IC"]==metrics_dict["IC"].max()]
        best2_model=metrics_dict["MAE"][metrics_dict["MAE"]==metrics_dict["MAE"].min()]
        print(f"For stock {ii}, I seemble model {best1_model.index[0]} and model {best2_model.index[0]} to obtain the final model ")
        signals_pd["assemble"]=(np.array(signals_pd[best1_model.index])+np.array(signals_pd[best2_model.index]))/2
        test_metric(["assemble"])

        print(f"------------------Current symbol is {ii}------------------")
        print(metrics_dict.to_string())
        print(f"-------------------------------------------------------")
        print("")

        # 画图
        plt.figure(dpi=400)
        plt.plot(all_dates[5:], signals_pd["grooundtruth"][5:], label="groundtruth")
        plt.plot(all_dates[5:], signals_pd["assemble"][5:], label="assembled model")
        # plt.plot(all_dates[20:], signals_pd["grooundtruth_mean10"][20:], label="grooundtruth_mean10")
        # plt.plot(all_dates[20:], signals_pd["grooundtruth_mean20"][20:], label="grooundtruth_mean20")
        for model_c in combine_method_name:
            plt.plot(all_dates[5:], signals_pd[model_c][5:], label=model_c,linewidth=1)
        plt.legend()
        plt.grid(True)
        plt.xticks(all_dates[5:][::10])
        plt.xticks(rotation=90)
        plt.title(f"{ii}")
        plt.tight_layout()
        plt.show()

        # choose best model
        plt.figure(dpi=400)
        plt.plot(all_dates[5:], signals_pd["grooundtruth"][5:], label="groundtruth")
        plt.plot(all_dates[5:], signals_pd["assemble"][5:], label="assembled model")
        plt.legend()
        plt.grid(True)
        plt.xticks(all_dates[5:][::10])
        plt.xticks(rotation=90)
        plt.title(f"{ii}")
        plt.tight_layout()
        plt.show()

        assert 1==2



        # for model_c in combine_method_name:
        #     plt.figure(dpi=400)
        #     plt.plot(all_dates, signals_pd["grooundtruth"], label="groundtruth")
        #     plt.plot(all_dates, signals_pd[model_c], label=model_c,linewidth=1)
        #     plt.legend()
        #     plt.grid(True)
        #     plt.xticks(all_dates[::12])
        #     plt.xticks(rotation=90)
        #     plt.tight_layout()
        #     plt.show()

        a=1