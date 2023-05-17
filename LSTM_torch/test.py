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

with open(r'statistic/robot_forward/hs_250_ls_1_sl_40/weights_vanilla_PID.pkl', 'rb') as f:
    df_w = pickle.load(f)

with open(r'statistic/robot_forward/hs_250_ls_1_sl_40/K_vanilla_PID.csv', 'rb') as f:
    K_df = pd.read_csv(f)

K_df_smooth = K_df.rolling(window=1000).mean()
df_w_smooth = df_w.rolling(window=1000).mean()
# fig, ax = plt.subplots(6, 1)
# for i in range(6):
#     ax[i].plot(K_df_smooth.iloc[:, i])

plt.plot(df_w_smooth.loc[:, 'loss_'])
plt.show()



