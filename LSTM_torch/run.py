# -*- coding: utf-8 -*- 
# @Time : 2023/2/7 15:00 
# @Author : Yinan 
# @File : run.py

import argparse
import torch
import lstm_train
import validation
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Input state stable LSTM')
parser.add_argument('--dataset', default='robot_forward', choices=['robot_forward', 'robot_forward', 'robot_inverse'], help='LSTM dataset')
parser.add_argument('--hidden_size', default=5, help='hidden size of LSTM')
if parser.parse_args().dataset == 'pHdata':
    input_size = output_size = 1
elif parser.parse_args().dataset == 'robot_inverse':
    input_size, output_size = 18, 6
elif parser.parse_args().dataset == 'robot_forward':
    input_size = output_size = 6
else:
    raise 'Nonexistent dataset!'
parser.add_argument('--input_size', default=input_size, help='input size of LSTM')
parser.add_argument('--output_size', default=output_size, help='output size of output layer')
parser.add_argument('--layers', default=1, help='number of layers of LSTM')
parser.add_argument('--batch_size', default=64, help='train batch size')
parser.add_argument('--epochs', default=80, help='maximum train epochs')
parser.add_argument('--tolerance', default=1e-6, help='minimum tolerance of loss')
parser.add_argument('--tol_stop', default=1e-10, help='minimum tolerance between 2 epochs')
parser.add_argument('--len_sequence', default=5, help='length of input sequence to LSTM')
parser.add_argument('--device', choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])

parser.add_argument(
    '--curriculum_learning', default='PID', choices=[None, '2zero', 'balance', 'exp', 'PID', 'IncrePID'], help='apply curriculum_learning or not')
parser.add_argument('--PID_coefficient', default=([10, 10], [0.5, 0.5], [0.1, 0.1]))
parser.add_argument('--reg_methode', default='vanilla', choices=['relu', 'log_barrier_BLS', 'vanilla'], help='regularization methode')
parser.add_argument('--gamma', default=torch.tensor([0., 0.]), help='value of gamma')
parser.add_argument('--threshold', default=torch.tensor([0.01, 0.05]), help='value of threshold')


if __name__ == '__main__':
    lstm_train.main(parser.parse_args())
    validation.main(parser.parse_args())


# TODO: 1. Norm_x have upper/lower bound, but hard to measure the value
#       2. adaptive K_p, K_i, K_d, manual selection is expensive:
#           2.1 based on the system dynamic: X^+ = f_LSTM(X, U), y = g_LSTM(X)
#               2.1.1 LSTM structure
#                     difficulty: it's a time variants system
#           2.2 a dynamic K, which will change during training process
#               2.2.1 treat K as a parameter instead of a hyperparameter,
#                     difficulty: may need to set an auxiliary loss function for K, i.e. Overshoot, response time ...


# python3 run.py [--dataset robot_forward] [--hidden_size] [--inputsize] [--output_size] [--layers] [--batch_size] [--epochs] [--tolerance] [--tol_stop] [--len_sequence] [--device 'cuda:1'] [--curriculum_learning]
# [--PID_coefficient] [--reg_methode] [--gamma] [--threshold]
