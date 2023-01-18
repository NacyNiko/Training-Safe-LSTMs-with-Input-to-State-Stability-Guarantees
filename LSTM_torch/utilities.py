# -*- coding: utf-8 -*- 
# @Time : 2022/12/21 16:36 
# @Author : Yinan 
# @File : utilities.py

import torch
import pandas as pd
import numpy as np


def normalize(arr):
    _max = np.max(arr)
    _min = np.min(arr)
    _range = _max - _min
    return (arr - _min) / _range


class DataCreater:
    # data path
    def __init__(self, train_x_path, train_y_path):
        self.path_x = train_x_path
        self.path_y = train_y_path

    def creat_new_dataset(self, seq_len=20):
        """ read original data """
        train_x = torch.tensor(np.array(pd.read_csv(self.path_x, index_col=0))).squeeze().to(torch.float32)

        train_x = (train_x - torch.mean(train_x)) / torch.std(train_x)   # normilization
        train_y = torch.tensor(np.array(pd.read_csv(self.path_y, index_col=0))).squeeze()

        """ create new data set """
        x_list = torch.empty([1, 2, seq_len+1])
        y_list = torch.empty([1, 1])

        l = len(train_x)-seq_len

        for i in range(l):
            """
            feature0: previous y[i:i + seq_len] and 0 at time step y[seq_len + i]
            feature1: input u[seq_len + i] at time step seq_len + i
            """
            feature0 = torch.cat([train_y[i:i + seq_len].unsqueeze(dim=0), torch.zeros([1, 1])], dim=1)
            feature1 = torch.cat([torch.zeros([1, seq_len]), train_x[i+seq_len].unsqueeze(dim=0).unsqueeze(dim=0)], dim=1)
            feature = torch.cat([feature0, feature1], dim=0).unsqueeze(0)
            x_list = torch.cat([x_list, feature], dim=0)
            y_list = torch.cat([y_list, train_y[i+seq_len].unsqueeze(dim=0).unsqueeze(dim=0)], dim=0)
        return x_list[1:, :, :], y_list[1:]


class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label

    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels

    def __len__(self):
        return len(self.data)


