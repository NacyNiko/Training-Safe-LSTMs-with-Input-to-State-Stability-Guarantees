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
        df = df.iloc[:, :]
        df = df.rolling(window=50).mean()

        fig, ax = plt.subplots(5, 1)
        for i, k in enumerate(['c1','c2','reg_loss1','reg_loss2','loss_']):
            # fig, ax = plt.subplots(1, 1)
            ax[i].plot(df.loc[:, k])
            ticks = np.linspace(0, df.shape[0], 6)  # 原始的刻度位置，比如0,1000,2000,...,5000
            labels = (ticks / df.shape[0] * 100).astype(int)
            ax[i].set_xticks(ticks)
            ax[i].set_xticklabels(labels)
            ax[i].set_xlabel('Epoch', fontsize=15)
            # ax[i].set_title('{} of {} strategy'.format('C1' if x == 'c1' else 'C2', 'Exponential' if cur == 'exp' else 'PID'))
        plt.show()


plot_C('robot_forward', 'exp', 'vanilla')