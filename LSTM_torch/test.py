# -*- coding: utf-8 -*- 
# @Time : 2022/12/12 16:20 
# @Author : Yinan 
# @File : test.py
import torch
from torch import nn
import pandas as pd
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('statistic/pHdata/weights_vanilla_PID.csv')
fig, ax = plt.subplots(7, 1, sharex=True)
colors = ['#8A2BE2', '#BA55D3', '#FF1493', '#00FFFF', '#A52A2A', '#7FFF00']
for i in range(df.shape[1]):
    if 0 <= i <= 5:
        # if i / 2:
        ax[0].plot(df.iloc[:7000, i], label=df.columns[i])
        ax[0].legend()
    else:
        ax[i-5].plot(df.iloc[:7000, i], label=df.columns[i], c=colors[i-6])
        ax[i-5].legend()
    ax[-1].set_xticks([*range(0, 7000, 700)], [*range(0, 100, 10)])
    ax[-1].set_xlabel('Epoch')
plt.show()





