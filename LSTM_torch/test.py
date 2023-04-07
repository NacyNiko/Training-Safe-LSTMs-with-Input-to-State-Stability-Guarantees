# -*- coding: utf-8 -*- 
# @Time : 2023/3/22 10:40 
# @Author : Yinan 
# @File : test.py

from utilities import PlotGraph
import pandas as pd
import matplotlib.pyplot as plt

# pg = PlotGraph('pHdata')
# pg.line_plot()
# pg.plot_K()

datay = pd.read_csv(r'../data/pHdata/train/train_output.csv')
prediction = pd.read_csv(r'../LSTM_torch/statistic/pHdata/Test_vanilla_PID.csv')
plt.plot(datay.iloc[:, 1])
plt.plot(prediction.iloc[1:, 0])
plt.show()