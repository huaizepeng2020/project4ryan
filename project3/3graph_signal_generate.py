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

os.environ['CASTLE_BACKEND'] = 'pytorch'

from castle.common import GraphDAG
from castle.metrics import MetricsDAG
from castle.datasets import DAG, IIDSimulation
from castle.algorithms import DAG_GNN, GOLEM, Notears


def parse_args():
    parser = argparse.ArgumentParser(description="data_download_from_fred")

    # ===== download ===== #
    parser.add_argument('--symbols', type=str, default="[ 'GOOG', 'AMZN', 'JPM', 'GME', 'XOM']",
                        help='concerned symbols')
    parser.add_argument('--symbol_target', type=str, default="SPY", help='target symbol')
    parser.add_argument('--period_IC', type=int, default=40, help='IC during future two months')

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
    symbols = eval(args.symbols)
    symbol_target = args.symbol_target
    period_IC = args.period_IC
    backtest_start = args.backtest_start
    backtest_end = args.backtest_end
    used_T = args.used_T
    retrain_T = args.retrain_T

    datas_return = pd.read_csv(f"datas/stocks_x_orginalreturn_{1}.csv", index_col=0)

    datas_new = pd.read_csv(f"datas/stocks_x_preprocess_3.csv", index_col=0)
    data_index = list(datas_new.index)
    all_dates = datas_new[(datas_new.index >= backtest_start) & (datas_new.index <= backtest_end)].index

    num_stocks = len(datas_new.columns)

    pnl_all = []
    IC_all = []
    time_all = []
    # for i in all_dates[::retrain_T]:
    # for i_idx,i in enumerate(all_dates[::time_interval]):
    for i_idx, i in enumerate(all_dates):
        time_all.append(i)
        end_date_idx_c = data_index.index(i)
        start_date_idx_c = end_date_idx_c - used_T

        # to avoid lokk-ahead information
        past_price_c = datas_new.iloc[start_date_idx_c - 3:end_date_idx_c - 3]

        # GOLEM based on NoTears
        gnn = GOLEM(num_iter=1e3, graph_thres=0.3)
        # gnn = Notears()
        gnn.learn(past_price_c)
        matrix_c = np.array(gnn.causal_matrix)

        with open(os.path.join("graphs") + f'/{i}_{3}.pickle', 'wb') as file:
            pickle.dump(matrix_c, file)

        # signal
        alpha_signal = matrix_c @ np.ones((num_stocks, 1)).squeeze(-1)
        alpha_signal = (alpha_signal - np.mean(alpha_signal)) / np.std(alpha_signal)
        with open(os.path.join("signals") + f'/{i}_{3}.pickle', 'wb') as file:
            pickle.dump(alpha_signal, file)

        print(f"graph and signal on {i} have been generated")
