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
    parser.add_argument('--my_api', type=str, default="34e95c57aae8485f53fef1a429cef662", help='API access of FRED')
    parser.add_argument('--target', type=str, default="CSUSHPINSA", help='the symbol of S&P CoreLogic Case-Shiller U.S. National Home Price Index')
    parser.add_argument('--feature', type=int, default=["house", "rent", "home", "family" and "income"], help='all features containing the keyword')
    parser.add_argument('--threshold_popularity', type=int, default=3, help='threshold of popularity')


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

    target_y=args.target
    feature_x=args.feature
    my_api=args.my_api
    threshold_popularity=args.threshold_popularity

    # FRED API
    fred = Fred(api_key=my_api)
    data_y = fred.get_series(target_y)
    all_data_x=[]
    for i in feature_x:
        data_x = fred.search(i,limit=5000,order_by="popularity",sort_order="desc",filter=("frequency","Monthly"))
        data_x["popularity"]=np.array(data_x["popularity"],dtype=np.float)
        data_x = data_x[data_x["popularity"]>=threshold_popularity]
        data_x = data_x[data_x["observation_start"]<=pd.to_datetime("1981-01-01")]
        all_data_x.append(data_x)
    all_data_x=pd.concat(all_data_x,axis=0)
    all_data_x_name=list(all_data_x["id"])
    data_xs={}
    for i in tqdm(all_data_x_name):
        if i!=target_y:
            data_xs[i]=fred.get_series(i)
            time.sleep(3)

    with open('datas/factors.pickle', 'wb') as file:
        pickle.dump(data_xs,file)
    data_y.to_csv("datas/target.csv")




