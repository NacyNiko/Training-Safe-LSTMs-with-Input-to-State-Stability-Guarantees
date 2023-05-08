# -*- coding: utf-8 -*- 
# @Time : 2022/12/5 14:25 
# @Author : Yinan 
# @File : LSTM.py
import copy

import numpy as np

from scipy.optimize import minimize
import pandas as pd
import pickle
import time
import datetime

def round_params(params, decimal_places=4):
    return np.round(params, decimal_places)

def sigmoid(x):
    return np.where(x >= 0, 1.0 / (1 + np.exp(-x)), np.exp(x) / (np.exp(x) + 1))

def tanh(x):
    return np.tanh(x)

def fun(W):
    myloss = 0

    u = data_input
    y_true = data_output

    C = np.zeros((numHiddenUnits, 1))
    X = np.zeros((numHiddenUnits, 1))

    # W = [W_f, W_i, W_o, W_c, U_f, U_i, U_o, U_c, b_f, b_i, b_o, b_c, W_y, b_y]
    W_f = W[0:10].reshape(5, 2)
    W_i = W[10:20].reshape(5, 2)
    W_o = W[20:30].reshape(5, 2)
    W_c = W[30:40].reshape(5, 2)
    U_f = W[40:65].reshape(5, 5)
    U_i = W[65:90].reshape(5, 5)
    U_o = W[90:115].reshape(5, 5)
    U_c = W[115:140].reshape(5, 5)
    b_f = W[140:145].reshape(5, 1)
    b_i = W[145:150].reshape(5, 1)
    b_o = W[150:155].reshape(5, 1)
    b_c = W[155:160].reshape(5, 1)
    W_y = W[160:165].reshape(5, 1)
    b_y = W[165]

    for i in range(u.shape[0]):
        C = sigmoid(np.dot(W_f, pd.DataFrame(u.iloc[i, :])) + np.dot(U_f, X) + b_f) * C + \
            sigmoid(np.dot(W_i, pd.DataFrame(u.iloc[i, :])) + np.dot(U_i, X) + b_i) * \
            tanh(np.dot(W_c, pd.DataFrame(u.iloc[i, :])) + np.dot(U_c, X) + b_c)
        X = sigmoid(np.dot(W_o, pd.DataFrame(u.iloc[i, :])) + np.dot(U_o, X) + b_o) * tanh(C)
        y_pred = np.dot(W_y.T, X) + b_y
        y_pred = y_pred * std_y + mean_y
        myloss += np.power((y_pred - y_true[i]).flatten(), 2)
    final_loss = myloss / len(u)

    final_loss = round_params(final_loss, 10)
    print(f'loss:{final_loss[0]:.10f}')
    return final_loss    # normalization

def constraint(W):
    W_f = W[0:10].reshape(5, 2)
    W_i = W[10:20].reshape(5, 2)
    W_o = W[20:30].reshape(5, 2)

    U_f = W[40:65].reshape(5, 5)
    U_i = W[65:90].reshape(5, 5)
    U_o = W[90:115].reshape(5, 5)
    U_c = W[115:140].reshape(5, 5)

    b_f = W[140:145].reshape(5, 1)
    b_i = W[145:150].reshape(5, 1)
    b_o = W[150:155].reshape(5, 1)

    con = np.array([1 - (1 + sigmoid(np.linalg.norm(np.hstack((W_o, U_o, b_o)), np.inf))) *
           sigmoid(np.linalg.norm(np.hstack((W_f, U_f, b_f)), np.inf)),
           1 - (1 + sigmoid(np.linalg.norm(np.hstack((W_o, U_o, b_o)), np.inf))) *
           sigmoid(np.linalg.norm(np.hstack((W_i, U_i, b_i)), np.inf)) * np.linalg.norm(U_c, 1)])

    return con


def initial():
    np.random.seed(0)
    Wf = np.random.randn(numHiddenUnits * (numInput + numOutput))   # 10
    Wi = np.random.randn(numHiddenUnits * (numInput + numOutput))   # 10
    Wo = np.random.randn(numHiddenUnits * (numInput + numOutput))   # 10
    Wc = np.random.randn(numHiddenUnits * (numInput + numOutput))   # 10
    Uf = np.random.randn(numHiddenUnits * numHiddenUnits)  # 25
    Ui = np.random.randn(numHiddenUnits * numHiddenUnits)  # 25
    Uc = np.random.randn(numHiddenUnits * numHiddenUnits)  # 25
    Uo = np.random.randn(numHiddenUnits * numHiddenUnits)  # 25
    bf = np.random.randn(numHiddenUnits * 1)   # 5
    bi = np.random.randn(numHiddenUnits * 1)   # 5
    bc = np.random.randn(numHiddenUnits * 1)   # 5
    bo = np.random.randn(numHiddenUnits * 1)   # 5
    Wy = np.random.randn(numHiddenUnits * numOutput)  # 5
    by = np.random.randn(1)  # 1

    return np.concatenate([Wf, Wi, Wo, Wc, Uf, Ui, Uo, Uc, bf, bi, bo, bc, Wy, by])


def load_data(train=True, noise=True):
    x_mean = pd.read_csv("../data/pHdata/train/train_input.csv", header=None).iloc[:, 1].mean(axis=0)
    x_std = pd.read_csv("../data/pHdata/train/train_input.csv", header=None).iloc[:, 1].std(axis=0)
    y_mean = pd.read_csv("../data/pHdata/train/train_output.csv", header=None).iloc[:, 1].mean(axis=0)
    y_std = pd.read_csv("../data/pHdata/train/train_output.csv", header=None).iloc[:, 1].std(axis=0)

    train_x = pd.read_csv("../data/pHdata/train/train_input.csv", header=None).iloc[:, 1]
    train_x = (train_x - x_mean) / x_std

    train_y = pd.read_csv("../data/pHdata/train/train_output.csv", header=None).iloc[:, 1]
    labels = copy.deepcopy(train_y)[1:]
    labels = labels.reset_index(drop=True)
    train_y = (train_y - y_mean) / y_std

    feature = pd.concat([train_y[:-1], train_x[:-1]], axis=1)

    return feature, labels, y_mean, y_std


""" hyperparameters """
noise, train = True, True
for if_con in [True]:
    print('start optimize with constraints' if if_con else 'start optimize without constraints')
    numInput = 1
    numOutput = 1
    numHiddenUnits = 5
    tol = 10-6

    """ training """
    W0 = initial()
    cons = ({'type': 'ineq', 'fun': constraint})

    data_input, data_output, mean_y, std_y = load_data()

    method_ = "SLSQP"
    time_start = time.time()
    options = {'maxiter': 10}

    res = minimize(fun, W0, method=method_, tol=tol, constraints=cons if if_con else None, options=options)

    time_end = time.time()

    res.running_time = time_end - time_start
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    res_file = open("models/model_{}_{}_{}_{}_{}.pkl".format("noise" if noise else "clean", method_, tol, 'w_con' if if_con else 'wo_con', now), "wb")
    pickle.dump(res, res_file)
    res_file.close()
    print('end optimize with constraints' if if_con else 'end optimize without constraints')