# -*- coding: utf-8 -*- 
# @Time : 2023/3/22 10:40 
# @Author : Yinan 
# @File : test.py

from utilities import PlotGraph
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np


# with open(r'statistic/robot_forward/hs_250_ls_1_sl_40/K_vanilla_PID.csv', 'rb') as f:
#     K_df = pd.read_csv(f)




def plot_C(dataset, cur, mat):
    with open('statistic/{}/{}/weights_{}_{}.pkl'.format(dataset
            , 'hs_5_ls_1_sl_10' if dataset == 'pHdata' else 'hs_250_ls_1_sl_40', mat, cur), 'rb') as f:
        df = pickle.load(f)
        df = df.rolling(window=5).min()

        for x in ['c1', 'c2']:
            fig, ax = plt.subplots(1, 1)
            ax.plot(df.loc[:, x])
            ticks = np.linspace(0, df.shape[0], 6)  # 原始的刻度位置，比如0,1000,2000,...,5000
            labels = (ticks / df.shape[0] * 100).astype(int)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.set_xlabel('Epoch', fontsize=15)
            ax.set_title('{} of {} strategy'.format('C1' if x == 'c1' else 'C2', 'Exponential' if cur == 'exp' else 'PID'))
            plt.show()

plot_C('robot_forward', 'PID', 'vanilla')