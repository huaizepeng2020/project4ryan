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
    parser.add_argument('--clean_way_4x', type=str, default="[1,3]",
                        help='encode of clean method for fatures, 1 for forward filling, 2 for back filling, 3 for zero filling')
    parser.add_argument('--clean_way_4y', type=str, default="[4]",
                        help='encode of clean method for targets, 1 for forward filling, 2 for back filling, 3 for zero filling, 4 for interpolation ')
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
    methods_code_4x = args.clean_way_4x
    methods_code_4x = eval(methods_code_4x)
    methods_code_4y = args.clean_way_4y
    methods_code_4y = eval(methods_code_4y)

    # load data
    with open('datas/factors.pickle', 'rb') as file:
        data_xs = pickle.load(file)
    data_y = pd.read_csv("datas/target.csv",index_col=0)

    data_x1 = pd.concat(data_xs, axis=1)
    print(f"---We have {len(data_x1.columns)} raw features.---")

    # clean inf and nan
    # We use the forward value to replace the inf and nan value and avoid look-ahead information
    # Then if there is still inf and nan value, we use zero to replace.

    data_x1.replace(np.inf, np.nan, inplace=True)
    data_x1.replace(-np.inf, np.nan, inplace=True)


    for i in methods_code_4x:
        data_x1 = clean_function(data_x1,i)

    for i in methods_code_4y:
        data_y = clean_function(data_y,i)

    # delete sample with nan target
    idx_valid=data_y[data_y.isin([np.nan, np.inf, -np.inf]).any(axis=1)].index
    data_x1=data_x1.drop(idx_valid)
    data_y=data_y.drop(idx_valid)

    # save
    data_x1.to_csv("datas/factors_clean.csv")
    data_y.to_csv("datas/target_clean.csv")