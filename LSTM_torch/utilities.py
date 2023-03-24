# -*- coding: utf-8 -*- 
# @Time : 2022/12/21 16:36 
# @Author : Yinan 
# @File : utilities.py
import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np


def normalize(arr):
    _max = np.max(arr)
    _min = np.min(arr)
    _range = _max - _min
    return (arr - _min) / _range


def cal_constraints(hidden_size, paras, df=None):
    columns = ['norm i', 'sigmoid norm i', 'norm o', 'sigmoid norm o'
        , 'norm f', 'sigmoid norm f', 'norm cell state', 'c1', 'c2']
    hidden_size = hidden_size
    parameters = list()
    for param in paras:
        parameters.append(param)
    weight_ih = parameters[0]
    weight_hh = parameters[1]
    bias = parameters[2] + parameters[3]

    W_o = weight_ih[-hidden_size:, :]
    U_o = weight_hh[-hidden_size:, :]

    b_o = bias[-hidden_size:].unsqueeze(1)

    W_f = weight_ih[hidden_size:2 * hidden_size, :]
    U_f = weight_hh[hidden_size:2 * hidden_size, :]
    b_f = bias[hidden_size:2 * hidden_size].unsqueeze(1)

    W_i = weight_ih[:hidden_size, :]
    U_i = weight_hh[:hidden_size, :]
    b_i = bias[:hidden_size].unsqueeze(1)

    U_c = weight_hh[2 * hidden_size: 3 * hidden_size, :]

    norm_ig = torch.norm(torch.hstack((W_i, U_i, b_i)), torch.inf)
    sigmoid_norm_ig = torch.sigmoid(norm_ig)
    norm_og = torch.norm(torch.hstack((W_o, U_o, b_o)), torch.inf)
    sigmoid_norm_og = torch.sigmoid(norm_og)
    norm_fg = torch.norm(torch.hstack((W_f, U_f, b_f)), torch.inf)
    sigmoid_norm_fg = torch.sigmoid(norm_fg)
    norm_cs = torch.norm(U_c, 1)

    con1 = (1 + sigmoid_norm_og) * sigmoid_norm_fg - 1
    con2 = (1 + sigmoid_norm_og) * sigmoid_norm_ig * norm_cs - 1

    if df is not None:
        df.loc[len(df.index), columns] = [norm_ig.item(), sigmoid_norm_ig.item(), norm_og.item(), sigmoid_norm_og.item(),
                                 norm_fg.item(), sigmoid_norm_fg.item(), norm_cs.item(), con1.item(), con2.item()]
    # TODO: show how does each term change.
    return [con1, con2], df
    # TODO: PID strategy


class DataCreater:
    # data path
    def __init__(self, train_x_path, train_y_path, test_x_path, test_y_path, input_size, output_size, train=True):
        self.train_x_path = train_x_path
        self.train_y_path = train_y_path
        self.test_x_path = test_x_path
        self.test_y_path = test_y_path
        self.input_size = input_size
        self.output_size = output_size
        self.train = train

        self.mean = torch.mean(torch.tensor(np.array(pd.read_csv(self.train_x_path, index_col=0))).to(torch.float32), dim=0)
        self.std = torch.std(torch.tensor(np.array(pd.read_csv(self.train_x_path, index_col=0))).to(torch.float32), dim=0)

    def creat_new_dataset(self, seq_len=20):
        """ read original data """

        train_x = torch.tensor(np.array(pd.read_csv(self.train_x_path if self.train else self.test_x_path, index_col=0))).to(torch.float32)

        train_x = (train_x - self.mean) / self.std   # normilization
        # TODO：only use mean/ std of training set, torch.mean
        train_y = torch.tensor(np.array(pd.read_csv(self.train_y_path if self.train else self.test_y_path , index_col=0)))

        """ create new data set """
        x_list = torch.empty([1, self.input_size + self.output_size, seq_len+1])   # instance, features, seq
        y_list = torch.empty([1, self.output_size])

        l = len(train_x)-seq_len

        for i in range(l):
            """
            feature0: previous y[i:i + seq_len] and 0 at time step y[seq_len + i]
            feature1: input u[seq_len + i] at time step seq_len + i
            """
            feature0 = torch.cat([train_y[i:i + seq_len, :].transpose(0, 1).unsqueeze(dim=0), torch.zeros([1,  self.output_size, 1])], dim=2)
            feature1 = torch.cat([torch.zeros([self.input_size, seq_len]), train_x[i+seq_len, :].unsqueeze(1)], dim=1).unsqueeze(dim=0)
            feature = torch.cat([feature0, feature1], dim=1)
            x_list = torch.cat([x_list, feature], dim=0)
            y_list = torch.cat([y_list, train_y[i+seq_len, :].unsqueeze(0)], dim=0)
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


class PlotGraph:
    def __init__(self, dataset):
        self.dataset = dataset
        self.path = './statistic/{}/weights_vanilla_PID.csv'.format(dataset)
        self.data = pd.read_csv(self.path)
        self.columns = 'norm i,sigmoid norm i,norm o,sigmoid norm o,norm f,sigmoid norm f,norm cell state,c1,c2,reg_loss1,reg_loss2,loss_'.split(',')

    def line_plot(self):
        fig, ax = plt.subplots(7, 1)

        for i in range(6):
            ax[0].plot(self.data.iloc[:, i], label=self.columns[i], c=(np.random.random(), np.random.random(), np.random.random()))
            ax[0].legend()

        for i in range(1, 7):
            ax[i].plot(self.data.iloc[:, i+5], label=self.columns[i+5], c=(np.random.random(), np.random.random(), np.random.random()))
            ax[i].legend()

        plt.show()
        plt.savefig('./statistic/{}/{}.fig'.format(self.data, self.data))


