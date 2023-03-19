# -*- coding: utf-8 -*- 
# @Time : 2022/12/9 17:39 
# @Author : Yinan 
# @File : LSTM_trian_ipm.py

import numpy as np
import pyipm
import torch
from scipy.optimize import minimize
import pandas as pd
import pickle
import time
from pyipm import IPM


def load_data(train=True, noise=True):
    if train:
        if noise:
            data_input = pd.read_csv("../data/train/train_input_noise.csv", header=None)
            data_output = pd.read_csv("../data/train/train_output_noise.csv", header=None)
        else:
            data_input = pd.read_csv("../data/train/train_input_clean.csv", header=None)
            data_output = pd.read_csv("../data/train/train_output_clean.csv", header=None)
    else:
        if noise:
            data_input = pd.read_csv("../data/val/val_input_noise.csv", header=None)
            data_output = pd.read_csv("../data/val/val_output_noise.csv", header=None)
        else:
            data_input = pd.read_csv("../data/val/val_input_clean.csv", header=None)
            data_output = pd.read_csv("../data/val/val_output_clean.csv", header=None)
    data_input = data_input.iloc[:, 1]
    data_output = data_output.iloc[:, 1]
    return data_input, data_output


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



def tanh(x):
    return np.tanh(x)

def fun(W):
    # W: (146,)
    myloss = 0

    u = data_input
    y_true = data_output

    C = np.zeros((numHiddenUnits, numInput))
    X = np.zeros((numHiddenUnits, numInput))

    # W = [W_f, W_i, W_o, W_c, U_f, U_i, U_o, U_c, b_f, b_i, b_o, b_c, W_y, b_y]
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

    for i in range(len(data_input)):
        C = sigmoid(np.dot(W_f, u[i]) + np.dot(U_f, X) + b_f) * C + \
            sigmoid(np.dot(W_i, u[i]) + np.dot(U_i, X) + b_i) * \
            tanh(np.dot(W_c, u[i]) + np.dot(U_c, X) + b_c)
        X = sigmoid(np.dot(W_o, u[i]) + np.dot(U_o, X) + b_o) * tanh(C)
        y_pred = np.dot(W_y.T, X) + b_y
        myloss += np.linalg.norm(y_pred - y_true[i])
    return myloss / len(u)    # normalization


def constraint(W):
    W_f = W[0:5].reshape(5, 1)
    W_i = W[5:10].reshape(5, 1)
    W_o = W[10:15].reshape(5, 1)

    U_f = W[20:45].reshape(5, 5)
    U_i = W[45:70].reshape(5, 5)
    U_o = W[70:95].reshape(5, 5)
    U_c = W[95:120].reshape(5, 5)

    b_f = W[120:125].reshape(5, 1)
    b_i = W[125:130].reshape(5, 1)
    b_o = W[130:135].reshape(5, 1)

    # try vector norm
    con = [(1 + sigmoid(np.linalg.norm(np.hstack((W_o, U_o, b_o)).flatten(), np.inf))) * \
           sigmoid(np.linalg.norm(np.hstack((W_f, U_f, b_f)).flatten(), np.inf)) - 1,
    (1 + sigmoid(np.linalg.norm(np.hstack((W_o, U_o, b_o)).flatten(), np.inf))) * \
           sigmoid(np.linalg.norm(np.hstack((W_i, U_i, b_i)).flatten(), np.inf)) * np.linalg.norm(U_c, 1) - 1]
    # except:
    #     print(W_o, U_o, b_o, W_f, U_f, b_i, W_o, U_o, b_o, W_f, U_f, b_i)
    #     print(W)
    return con



def initial():
    np.random.seed(10)
    Wf = np.random.randn(numHiddenUnits * numInput)
    Wi = np.random.randn(numHiddenUnits * numInput)
    Wo = np.random.randn(numHiddenUnits * numInput)
    Wc = np.random.randn(numHiddenUnits * numInput)
    Uf = np.random.randn(numHiddenUnits * numHiddenUnits)
    Ui = np.random.randn(numHiddenUnits * numHiddenUnits)
    Uc = np.random.randn(numHiddenUnits * numHiddenUnits)
    Uo = np.random.randn(numHiddenUnits * numHiddenUnits)
    bf = np.random.randn(numHiddenUnits * 1)
    bi = np.random.randn(numHiddenUnits * 1)
    bc = np.random.randn(numHiddenUnits * 1)
    bo = np.random.randn(numHiddenUnits * 1)
    Wy = np.random.randn(numHiddenUnits * numOutput)
    by = np.random.randn(1)

    return np.concatenate([Wf, Wi, Wo, Wc, Uf, Ui, Uo, Uc, bf, bi, bo, bc, Wy, by])


data_input, data_output = load_data()
numInput = 1
numOutput = 1
numHiddenUnits = 5
W0 = initial()


cons = ([{'type': 'ineq', 'fun': constraint}])

""" training """
time_start = time.time()
problem = pyipm.IPM(x0=W0, f=fun, ci=constraint)
weight, s, lda, fval, kkt = problem.solve(x0=W0)
time_end = time.time()

running_time = time_end - time_start

res_file = open("pyipm.pkl", "wb")
pickle.dump(weight, res_file)
res_file.close()

