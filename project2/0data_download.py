import numpy as np
import pandas as pd
import random
import scipy
import scipy.stats
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

    for i in symbols:
        datas = []
        data_c=pd.read_csv(f"datas/{i}.csv")[["Date","Adj Close"]].rename(columns={'Adj Close':i})
        data_c1=pd.read_csv(f"datas/{symbol_target}.csv")[["Date","Adj Close"]].rename(columns={'Adj Close':symbol_target})
        datas=pd.merge(data_c,data_c1)
        IC_all=[]
        for j in range(len(datas)-period_IC):
            IC_c=scipy.stats.pearsonr(datas.iloc[j:j+period_IC+1][i],datas.iloc[j:j+period_IC+1][symbol_target])[0]
            IC_all.append(IC_c)
        datas_new=datas.iloc[:len(datas)-period_IC]
        datas_new["IC"]=np.array(IC_all)
        datas_new.to_csv(f"datas/{i}_xy.csv")




