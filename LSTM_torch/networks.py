# -*- coding: utf-8 -*- 
# @Time : 2023/3/19 8:57 
# @Author : Yinan 
# @File : networks.py
from torch import nn
import torch


class LstmRNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=5, output_size=1, num_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  #  [seq_len, batch_size, input_size] --> [seq_len, batch_size, hidden_size]
        self.linear1 = nn.Linear(hidden_size, output_size)  #  [seq_len, batch_size, hidden_size] --> [seq_len, batch_size, output_size]

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = x.view(s, b, -1)
        return x[-1, :, :]


class PidNN(nn.Module):
    def __init__(self, input_size, hidden_size=5, output_size=6):
        super(PidNN, self).__init__()
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        # self.sequential = nn.Sequential(
        #     nn.Linear(input_size, hidden_size)
        #     , nn.Tanh()
        #     , nn.Linear(hidden_size, output_size)
        # )

    def forward(self, x):
        x = self.linear1(x)
        x = torch.tanh(x)
        x = self.linear2(x)
        x = 50 * (1 + torch.tanh(x))
        out = x.reshape(3, 2)
        return out


