# -*- coding: utf-8 -*- 
# @Time : 2022/12/13 14:41 
# @Author : Yinan 
# @File : validation.py
import torch
from test import IssLSTM
from torch import nn
import pandas as pd
import torch.utils.data as Data
import matplotlib.pyplot as plt


model = IssLSTM()
# model.eval()
model.load_state_dict(torch.load('model.pth'))
u = pd.read_csv(r'../ISS_LSTM_by_ipm/train/train_input_noise.csv', header=None).iloc[:, 1]
y = pd.read_csv(r'../ISS_LSTM_by_ipm/train/train_output_noise.csv', header=None).iloc[:, 1]


""" load training set """
train_data = Data.TensorDataset(
    torch.FloatTensor(u),
    torch.FloatTensor(y)
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=1, shuffle=False, drop_last=False)
y_pred = []
hidden_prev = torch.randn(1, 5)
cell_prev = torch.randn(1, 5)
with torch.no_grad():
    for x, label in train_loader:
        y_hat, hidden_prev, cell_prev = model(x.unsqueeze(0), hidden_prev, cell_prev)
        y_pred.append(y_hat)


# loss = nn.MSELoss()
# mse = loss(y, y_pred)




plt.plot(range(4400), y_pred)
plt.plot(range(4400), y)
plt.show()
