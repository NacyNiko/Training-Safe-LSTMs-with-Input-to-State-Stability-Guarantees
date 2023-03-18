# -*- coding: utf-8 -*- 
# @Time : 2023/2/7 15:00 
# @Author : Yinan 
# @File : run.py

import argparse
import torch
import lstm_train
import validation


parser = argparse.ArgumentParser(description='Input state stable LSTM')
parser.add_argument('--device', default='cuda:0', choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
parser.add_argument('--dataset', default='robot_forward', choices=['pHdata', 'robot_forward', 'robot_inverse'], help='LSTM dataset')
parser.add_argument('--hidden_size', default=5, help='hidden size of LSTM', type=int)
# if parser.parse_args().dataset == 'pHdata':
#     input_size = output_size = 1
# elif parser.parse_args().dataset == 'robot_inverse':
#     input_size, output_size = 18, 6
# elif parser.parse_args().dataset == 'robot_forward':
#     input_size = output_size = 6
# else:
#     raise 'Nonexistent dataset!'
parser.add_argument('--input_size', default=6, help='input size of LSTM', type=int)
parser.add_argument('--output_size', default=6, help='output size of output layer', type=int)
parser.add_argument('--layers', default=1, help='number of layers of LSTM', type=int)
parser.add_argument('--batch_size', default=64, help='train batch size', type=int)
parser.add_argument('--epochs', default=100, help='maximum train epochs', type=int)
parser.add_argument('--tolerance', default=1e-6, help='minimum tolerance of loss', type=float)
parser.add_argument('--tol_stop', default=1e-10, help='minimum tolerance between 2 epochs', type=float)
parser.add_argument('--len_sequence', default=5, help='length of input sequence to LSTM', type=int)


parser.add_argument(
    '--curriculum_learning', default='PID', choices=[None, '2zero', 'balance', 'exp', 'PID', 'IncrePID'], help='apply curriculum_learning or not')
parser.add_argument('--PID_coefficient', default=([10, 12.5], [0.5, 1], [0.1, 0.1]), type=tuple)
parser.add_argument('--reg_methode', default='vanilla', choices=['relu', 'log_barrier_BLS', 'vanilla'], help='regularization methode')
parser.add_argument('--gamma', default=torch.tensor([0., 0.]), help='value of gamma', type=torch.Tensor)
parser.add_argument('--threshold', default=torch.tensor([0.01, 0.05]), help='value of threshold', type=torch.Tensor)


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
#           2.3 Iterative method:
#               1. Initialize K_0 for PID controller, set max iterative num p_max
#               2. for epoch i < max epoch, for p <= p_max and plant_i, do
#                   2.1 train K_i^* with NN-PID
#                   2.2 update K_i^p = alpha * K_i^p + (1-alpha) * K_i^*
#                   2.3 p = p+1
#               3. train weights of plant_(i+1) with K_i^(p_max)
#               4. epoch = epoch + 1, set K_(i+1) = K_i^p_max, back to 2.

# python run.py --dataset robot_forward --hidden_size --inputsize --output_size --layers --batch_size --epochs --tolerance --tol_stop --len_sequence --device 'cuda:3' --curriculum_learning --PID_coefficient --reg_methode --gamma --threshold
