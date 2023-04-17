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

        self.mean_y = torch.mean(torch.tensor(np.array(pd.read_csv(self.train_y_path, index_col=0))).to(torch.float32), dim=0)

    def creat_new_dataset(self, seq_len):
        """ read original data """

        train_x = torch.tensor(np.array(pd.read_csv(self.train_x_path if self.train else self.test_x_path, index_col=0))).to(torch.float32)

        train_x = (train_x - self.mean) / self.std   # normilization
        # TODO：only use mean/ std of training set, torch.mean
        train_y = torch.tensor(np.array(pd.read_csv(self.train_y_path if self.train else self.test_y_path , index_col=0)))

        """ create new data set """
        feature = torch.cat([train_y[:-1, :], train_x[:-1, :]], dim=1)
        labels = train_y[seq_len:, :]

        return feature, labels


class GetLoader(torch.utils.data.Dataset):
    def __init__(self, data_root, data_label, seq_len, train: bool):
        self.data = data_root
        self.label = data_label
        self.seq_len = seq_len
        self.train = train

    def __getitem__(self, index):
        data = self.data[index:index + self.seq_len, :]
        labels = self.label[index, :]
        return data, labels

    def __len__(self):
        return len(self.label[:, 0])


class PlotGraph:
    def __init__(self, dataset):
        self.dataset = dataset
        self.path = './statistic/{}/weights_vanilla_PID.csv'.format(dataset)
        self.data = pd.read_csv(self.path)
        self.columns = 'norm i,sigmoid norm i,norm o,sigmoid norm o,norm f,sigmoid norm f,norm cell state,c1,c2,reg_loss1,reg_loss2,loss_'.split(',')
        self.K = './statistic/{}/K_vanilla_PID.csv'.format(dataset)

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

    def plot_K(self):
        fig, ax = plt.subplots(6, 1)

        for i in range(6):
            ax[i].plot(pd.read_csv(self.K).iloc[:, i],
                       c=(np.random.random(), np.random.random(), np.random.random()))
            ax[i].legend()

        plt.show()
        plt.savefig('./statistic/{}/{} K.fig'.format(self.data, self.data))


########### new ###############
class SaveLoss:
    def __init__(self, target: torch.tensor, window: int):
        self.loss = torch.tensor([[0, 0]])
        self.overshoot = torch.tensor([0, 0])
        self.response = torch.tensor([0, 0])
        self.steady_error = torch.tensor([0, 0])
        self.target = target
        self.window = window

    def add_loss(self, loss: torch.tensor, epoch: int):
        self.loss = torch.cat([self.loss, loss], dim=0)
        if self.loss.shape[0] - 1 >= self.window:
            for i in range(2):
                # if epoch % self.window == 0:
                self.overshoot = torch.tensor([max(max(self.loss[-self.window:][0] - self.target[0]), 0)
                    , max(max(self.loss[1][-self.window:] - self.target[0]), 0)])
                self.response = torch.tensor([torch.argmax(self.loss[-self.window:][0])
                    , torch.argmax(self.loss[-self.window:][1])])
                self.steady_error = torch.tensor([self.loss[-1][0] - self.target[0]
                    , self.loss[-1][1] - self.target[1]])
                return self.overshoot, self.response, self.steady_error
        else:
            return None, None, None

