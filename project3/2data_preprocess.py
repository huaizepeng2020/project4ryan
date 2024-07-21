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

    parser.add_argument('--return_period', type=str, default="[1,2,3]", help='IC during future two months')

    parser.add_argument('--clean_way_4x', type=str, default="[1,3]",
                        help='encode of clean method for fatures, 1 for forward filling, 2 for back filling, 3 for zero filling')

    parser.add_argument('--selection_method', type=str, default="[1,2]",
                        help='encode of selection method for fatures, 1 for IC test, 2 for correlation test')
    parser.add_argument('--selection_haperparameters', type=str, default="[[0.1],[0.7]]",
                        help='encode of clean method for targets, 1 for forward filling, 2 for back filling, 3 for zero filling, 4 for interpolation ')
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
    symbols=eval(args.symbols)
    symbol_target=args.symbol_target
    period_IC=args.period_IC
    selection_method = args.selection_method
    selection_method = eval(selection_method)
    selection_para = args.selection_haperparameters
    selection_para = eval(selection_para)
    return_period=args.return_period
    return_period=eval(return_period)

    datas_new = pd.read_csv(f"datas/stocks_x_clean.csv",index_col=0)

    for return_period_c in return_period:
        datas_new_c=datas_new.shift(-return_period_c)/datas_new-1
        datas_new_norm=datas_new_c.apply(lambda x:(x-x.mean())/x.std(),axis=1)

        datas_new_c.to_csv(f"datas/stocks_x_orginalreturn_{return_period_c}.csv")
        datas_new_norm.to_csv(f"datas/stocks_x_preprocess_{return_period_c}.csv")