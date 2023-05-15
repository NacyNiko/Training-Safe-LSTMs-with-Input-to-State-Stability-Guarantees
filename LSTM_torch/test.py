# -*- coding: utf-8 -*- 
# @Time : 2023/3/22 10:40 
# @Author : Yinan 
# @File : test.py

from utilities import PlotGraph
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# pg = PlotGraph('pHdata')
# pg.line_plot()
# pg.plot_K()

with open(r'./statistic/robot_forward/hs_150_ls_2_sl_40/weights_vanilla_2zero.pkl', 'rb') as f:
    df = pickle.load(f)

plt.plot(df.loc[1000:, 'reg_loss2'])
# plt.plot(df.iloc[1:, 0])
plt.show()