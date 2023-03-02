# -*- coding: utf-8 -*- 
# @Time : 2023/2/27 20:35 
# @Author : Yinan 
# @File : temp.py
import pandas as pd
import matplotlib.pyplot as plt

train = pd.read_csv('../data/robot_forward/train/train_input.csv', index_col=0)
val = pd.read_csv('../data/robot_forward/val/val_input.csv', index_col=0)
labels_train = pd.read_csv('../data/robot_forward/train/train_output_o.csv', index_col=0)

# fig, ax = plt.subplots(6, 1, figsize=(10, 10))
# for df in [val]:
#     for i in range(6):
#         ax[i].plot(df.iloc[:, i])
#
#     plt.show()

# train.iloc[:30000, :].to_csv('../data/robot_forward/train/train_input_split.csv')
# train.iloc[30000:, :].to_csv('../data/robot_forward/val/val_output_split.csv')
labels_train.iloc[:30000, :].to_csv('../data/robot_forward/train/train_output.csv')
labels_train.iloc[30000:, :].to_csv('../data/robot_forward/val/val_output.csv')