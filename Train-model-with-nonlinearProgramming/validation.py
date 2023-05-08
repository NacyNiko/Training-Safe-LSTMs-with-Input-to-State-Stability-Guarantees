# -*- coding: utf-8 -*- 
# @Time : 2022/12/7 20:56 
# @Author : Yinan 
# @File : Validation.py
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

""" define sigmoid """
def sigmoid(x):
    x_ravel = x.ravel()
    length = len(x_ravel)
    y = []
    for index in range(length):
        if x_ravel[index] >= 0:
            y.append(1.0 / (1 + np.exp(-x_ravel[index])))
        else:
            y.append(np.exp(x_ravel[index]) / (np.exp(x_ravel[index]) + 1))
    return np.array(y).reshape(x.shape)


""" define tanh """
def tanh(x):
    return np.tanh(x)

""" validation process """
def validate(W, data_input):
    """ LSTM """
    W_f = W[0:5].reshape(5, 1)
    W_i = W[5:10].reshape(5, 1)
    W_o = W[10:15].reshape(5, 1)
    W_c = W[15:20].reshape(5, 1)
    U_f = W[20:45].reshape(5, 5)
    U_i = W[45:70].reshape(5, 5)
    U_o = W[70:95].reshape(5, 5)
    U_c = W[95:120].reshape(5, 5)
    b_f = W[120:125].reshape(5, 1)
    b_i = W[125:130].reshape(5, 1)
    b_o = W[130:135].reshape(5, 1)
    b_c = W[135:140].reshape(5, 1)
    W_y = W[140:145].reshape(5, 1)
    b_y = W[145]

    u = data_input
    y_val = []


    """ initialize hidden state & cell state"""
    C = np.random.randn(5, 1)
    X = np.random.randn(5, 1)

    con = [-1 + (1 + sigmoid(np.linalg.norm(np.hstack((W_o, U_o, b_o)), np.inf))) *
           sigmoid(np.linalg.norm(np.hstack((W_f, U_f, b_f)), np.inf)),
           -1 + (1 + sigmoid(np.linalg.norm(np.hstack((W_o, U_o, b_o)), np.inf))) *
           sigmoid(np.linalg.norm(np.hstack((W_i, U_i, b_i)), np.inf)) * np.linalg.norm(U_c, 1)]

    for i in range(len(data_input)):
        C = sigmoid(np.dot(W_f, u.iloc[i]) + np.dot(U_f, X) + b_f) * C + \
            sigmoid(np.dot(W_i, u.iloc[i]) + np.dot(U_i, X) + b_i) * \
            tanh(np.dot(W_c, u.iloc[i]) + np.dot(U_c, X) + b_c)
        X = sigmoid(np.dot(W_o, u.iloc[i]) + np.dot(U_o, X) + b_o) * tanh(C)
        y_pred = np.dot(W_y.T, X) + b_y
        y_val.append(y_pred.squeeze())
    return y_val, con


""" define FIT as metric """
def FIT(y_hat, y):
    return 100 * (1 - np.linalg.norm(y_hat - y) / np.linalg.norm(y))

def nrmse(y, y_hat):  # normalization to y
    # return np.sqrt(1 / len(y)) * np.linalg.norm(y - y_hat)
    y_hat = torch.tensor(y_hat.values)
    y = torch.tensor(np.array(y))
    std = torch.std(y_hat, dim=0)
    norm = torch.norm(y.squeeze() - y_hat.squeeze())
    res = 1 / std * np.sqrt(1 / y.shape[0]) * norm
    return res
def main():
    """ load trained model """
    models = os.listdir('./models')
    for file in models:
        model_path = './models/' + file
        model_file = open(model_path, "rb")
        model = pickle.load(model_file)
        W = model.x


        f, ax = plt.subplots(2, 1)
        i = 0
        """ train or validation """
        for train in [True, False]:
            noise = True
            if train:
                """ load training set: 4400 samples """
                if noise:
                    data_input = pd.read_csv("../data/pHdata/train/train_input.csv", header=None).iloc[:, 1]
                    y_true = pd.read_csv("../data/pHdata/train/train_output.csv", header=None).iloc[:, 1]
                else:
                    data_input = pd.read_csv("../data/train/train_input_clean.csv", header=None).iloc[:, 1]
                    y_true = pd.read_csv("../data/train/train_output_clean.csv", header=None).iloc[:, 1]
            else:
                """ load validation set: 2250 samples """
                if noise:
                    data_input = pd.read_csv("../data/pHdata/val/val_input.csv", header=None).iloc[:, 1]
                    y_true = pd.read_csv("../data/pHdata/val/val_output.csv", header=None).iloc[:, 1]
                else:
                    data_input = pd.read_csv("../data/val/val_input_clean.csv", header=None).iloc[:, 1]
                    y_true = pd.read_csv("../data/val/val_output_clean.csv", header=None).iloc[:, 1]

            """ calculate y_val """
            y_val, cons = validate(W, data_input)

            """ plot """
            nrmse_score = nrmse(y_val[10:], y_true[10:])
            f.suptitle('c1:{} c2:{}'.format(cons[0], cons[1]))
            ax[i].plot(y_val[10:], color='m', label='pred', alpha=0.8)
            ax[i].plot(y_true[10:], color='c', label='real', linestyle='--', alpha=0.5)
            ax[i].tick_params(labelsize=5)
            ax[i].legend(loc='best')
            ax[i].set_title('NRMSE on {} set: {:.3f}'.format('train' if train else 'val', nrmse_score), fontsize=8)
            i += 1
        plt.savefig('./result/python/fig/{}.jpg'.format(file[:-4]), bbox_inches='tight', dpi=500)
            # plt.figure()
            # plt.title('constraints: c1:{}, c2:{}'.format(cons[0], cons[1]))
            # plt.plot([*range(len(data_input) - 100)], y_val[100:], color='b', label='Prediction')
            # plt.plot([*range(len(data_input) - 100)], y_true[100:], color='r', label='Ground Truth')
            #
            # # calculate FIT
            # fit_ = FIT(y_val[100:], y_true[100:])
            # plt.legend()
            # plt.title('pH on {} set with FIT: {:.3f}%'.format('train' if train else "validation", fit_))
            # plt.show()
if __name__ == '__main__':
    main()


