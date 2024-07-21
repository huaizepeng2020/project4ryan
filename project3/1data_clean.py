import numpy as np
import pandas as pd
import random
import torch
import time
from prettytable import PrettyTable
from tqdm import tqdm
import pickle
import gc
import argparse
from fredapi import Fred


# time.sleep(10)
def parse_args():
    parser = argparse.ArgumentParser(description="data_download_from_fred")

    # ===== download ===== #
    parser.add_argument('--symbols', type=str, default="[ 'GOOG', 'AMZN', 'JPM', 'GME', 'XOM']", help='concerned symbols')
    parser.add_argument('--symbol_target', type=str, default="SPY", help='target symbol')
    parser.add_argument('--period_IC', type=int, default=40, help='IC during future two months')

    parser.add_argument('--clean_way_4x', type=str, default="[1,3]",
                        help='encode of clean method for fatures, 1 for forward filling, 2 for back filling, 3 for zero filling')

    return parser.parse_args()

def clean_function(data_c,method):
    if method == 1:
        data_c = data_c.fillna(method='ffill')
    if method == 2:
        data_c = data_c.fillna(method='bfill')
    if method == 3:
        data_c = data_c.fillna(value=0)
    if method == 4:
        data_c = (data_c.fillna(limit=1,method='ffill')+data_c.fillna(limit=1,method='bfill'))/2
    return data_c

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

    methods_code_4x = args.clean_way_4x
    methods_code_4x = eval(methods_code_4x)

    datas_new=pd.read_csv(f"datas/stocks_x.csv")
    datas_new.replace(np.inf, np.nan, inplace=True)
    datas_new.replace(-np.inf, np.nan, inplace=True)

    for j in methods_code_4x:
        datas_new = clean_function(datas_new, j)
    datas_new.set_index("Date", inplace=True)
    datas_new.to_csv(f"datas/stocks_x_clean.csv")