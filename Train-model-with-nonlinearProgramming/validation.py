# -*- coding: utf-8 -*- 
# @Time : 2022/12/7 20:56 
# @Author : Yinan 
# @File : Validation.py
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    for i in range(len(data_input)):
        C = sigmoid(np.dot(W_f, u.iloc[i]) + np.dot(U_f, X) + b_f) * C + \
            sigmoid(np.dot(W_i, u.iloc[i]) + np.dot(U_i, X) + b_i) * \
            tanh(np.dot(W_c, u.iloc[i]) + np.dot(U_c, X) + b_c)
        X = sigmoid(np.dot(W_o, u.iloc[i]) + np.dot(U_o, X) + b_o) * tanh(C)
        y_pred = np.dot(W_y.T, X) + b_y
        y_val.append(y_pred.squeeze())
    return y_val


""" define FIT as metric """
def FIT(y_hat, y):
    return 100 * (1 - np.linalg.norm(y_hat - y) / np.linalg.norm(y))


def main():
    """ load trained model """
    model_file = open(r"models/model_noise_SLSQP_1e-10_wo_con_2023-01-16-00_54_08.pkl", "rb")
    model = pickle.load(model_file)
    W = model.x

    """ train or validation """
    train = True
    noise = True
    if train:
        """ load training set: 4400 samples """
        if noise:
            data_input = pd.read_csv("../data/train/train_input_noise.csv", header=None).iloc[:, 1]
            y_true = pd.read_csv("../data/train/train_output_noise.csv", header=None).iloc[:, 1]
        else:
            data_input = pd.read_csv("../data/train/train_input_clean.csv", header=None).iloc[:, 1]
            y_true = pd.read_csv("../data/train/train_output_clean.csv", header=None).iloc[:, 1]
    else:
        """ load validation set: 2250 samples """
        if noise:
            data_input = pd.read_csv("../data/val/val_input_noise.csv", header=None).iloc[:, 1]
            y_true = pd.read_csv("../data/val/val_output_noise.csv", header=None).iloc[:, 1]
        else:
            data_input = pd.read_csv("../data/val/val_input_clean.csv", header=None).iloc[:, 1]
            y_true = pd.read_csv("../data/val/val_output_clean.csv", header=None).iloc[:, 1]



    """ calculate y_val """
    y_val = validate(W, data_input)

    """ plot """
    plt.figure()
    plt.plot([*range(len(data_input) - 100)], y_val[100:], color='b', label='Prediction')
    plt.plot([*range(len(data_input) - 100)], y_true[100:], color='r', label='Ground Truth')

    # calculate FIT
    fit_ = FIT(y_val[100:], y_true[100:])
    plt.legend()
    plt.title('pH on {} set with FIT: {:.3f}%'.format('train' if train else "validation", fit_))
    plt.show()


if __name__ == '__main__':
    main()


