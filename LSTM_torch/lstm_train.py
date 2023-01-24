# -*- coding: utf-8 -*- 
# @Time : 2022/12/22 1:01 
# @Author : Yinan 
# @File : lstm_train.py

import torch
from torch import nn
import matplotlib.pyplot as plt
from utilities import DataCreater, GetLoader
from torch.utils.data import DataLoader
import numpy as np
import os
"""
输入： 2 维度，[t-w:t]的y 以及 [t]的u
输出： lstm最后一层隐状态经过Linear层的值

训练结果记录：
    1. 神经元取10， 序列长度5， 最终误差1e-4时过拟合
    2. 神经元取5， 序列长度5， 最终误差1e-3时欠拟合
    3. 神经元取5， 序列长度10， 最终误差1e-3时欠拟合
    4. 神经元取5， 序列长度5， 最终误差1e-4时 结果好
    5. 神经元取5， 序列长度5， 最终误差1e-5时 结果好
    
"""


# Define LSTM Neural Networks
class LstmRNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=5, output_size=1, num_layers=1, mean=-10., std=4.):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  #  [seq_len, batch_size, input_size] --> [seq_len, batch_size, hidden_size]
        init_dic = torch.load('./models/model_sl_5_bs_64_hs_5_ep_100_tol_1e-05_r_tensor([2., 2.])_thd_tensor([1., 1.]).pth')
        self.lstm.load_state_dict(init_dic)
        self.linear1 = nn.Linear(hidden_size, output_size)  #  [seq_len, batch_size, hidden_size] --> [seq_len, batch_size, output_size]

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = x.view(s, b, -1)
        return x[-1, :, :]


class IssLstmTrainer:
    def __init__(self, seq_len=5, input_size=2, hidden_size=5, output_size=1, num_layer=1, batch_size=64, max_epochs=100, tol=1e-5, ratio=[(2,2)], threshold=[(1,1)]):
        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.tol = tol
        self.ratio = torch.tensor(ratio)
        self.threshold = torch.tensor(threshold)
        self.old_model = None
        self.temp_lstm = LstmRNN().lstm

    @staticmethod
    def nrmse(y, y_hat):  # normalization to y
        y = y.squeeze(1)
        return np.sqrt(1 / y.shape[0]) * torch.norm(y - y_hat)

    @staticmethod
    def regularization_term(cons, r, threshold):  # paras = model.lstm.parameters()  [W, U, b1, b2]
        con1, con2 = cons[0], cons[1]
        return r[0] * torch.relu(con1 + threshold[0]) + r[1] * torch.relu(con2 + threshold[1])

    @staticmethod
    def log_barrier(cons, t=0.05):
        barrier = 0
        for con in cons:
            if con < 0:
                barrier += - (1/t) * torch.log(-con)
            else:
                return torch.tensor(float('inf'))
        return barrier

    def backtracking_line_search(self, params, alpha=0.1, typ='normal'):
        old_paras = self.flatten_params(self.old_model.lstm.parameters())
        new_paras = params
        cons = self.constraints(self.write_flat_params(self.temp_lstm, new_paras))
        ls = 0
        bar_loss = self.log_barrier(cons)
        while torch.isinf(bar_loss):
            new_paras = alpha * old_paras + (1-alpha) * new_paras
            cons = self.constraints(self.write_flat_params(self.temp_lstm, new_paras))
            bar_loss = self.log_barrier(cons)
            ls += 1
            if ls == 100:
                print('maximum search times reached')
                return bar_loss, new_paras
        # self.old_model = new_model.load_state_dict(new_paras)
        return bar_loss, new_paras

    def flatten_params(self, params):
        views = []
        for p in params:
            view = p.flatten(0)
            views.append(view)
        return torch.cat(views, 0)

    def write_flat_params(self, lstm, f_params):
        W_ih = f_params[:40].resize(20, 2)
        W_hh = f_params[40:140].resize(20, 5)
        bias1 = f_params[140:160].resize(20,)
        bias2 = f_params[160:].resize(20,)
        p_list = [W_ih, W_hh, bias1, bias2]
        i = 0
        for p in lstm.parameters():
            p.data = p_list[i]
            i += 1
        return lstm.parameters()

    def constraints(self, paras):
        hidden_size = self.hidden_size
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

        con1 = (1 + torch.sigmoid(torch.norm(torch.hstack((W_o, U_o, b_o)), torch.inf))) * \
               torch.sigmoid(torch.norm(torch.hstack((W_f, U_f, b_f)), torch.inf)) - 1
        con2 = (1 + torch.sigmoid(torch.norm(torch.hstack((W_o, U_o, b_o)), torch.inf))) * \
               torch.sigmoid(torch.norm(torch.hstack((W_i, U_i, b_i)), torch.inf)) * torch.norm(U_c, 1) - 1
        return [con1, con2]

    def train_begin(self, device='cuda:0' if torch.cuda.is_available() else 'cpu', reg_methode='vanilla'):
        if device == 'cuda:0':
            print('Training on GPU.')
        else:
            print('No GPU available, training on CPU.')

        for r in self.ratio:
            for thd in self.threshold:
                """ data set """
                data = [r'../data/train/train_input_noise.csv', r'../data/train/train_output_noise.csv']
                train_x, train_y = DataCreater(data[0], data[1]).creat_new_dataset(seq_len=self.seq_len)
                train_set = GetLoader(train_x, train_y)
                train_set = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=2)
                init_mean, init_std = -10., 4.
                # ----------------- train -------------------
                lstm_model = LstmRNN(self.input_size, self.hidden_size, output_size=self.output_size, num_layers=self.num_layer)

                while torch.isinf(self.log_barrier(self.constraints(lstm_model.lstm.parameters()))):
                    print('initial weight does not satisfy')
                    lstm_model = LstmRNN(mean=init_mean-1, std=init_std-0.5)

                self.old_model = lstm_model
                criterion = nn.MSELoss()
                optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)

                lstm_model.to(device)
                criterion.to(device)
                print('LSTM model:', lstm_model)
                print('model.parameters:', lstm_model.parameters)
                break_flag = False
                loss_prev = None
                for epoch in range(self.max_epochs):
                    for batch_cases, labels in train_set:
                        batch_cases = batch_cases.transpose(0, 1)
                        batch_cases = batch_cases.transpose(0, 2).to(torch.float32).to(device)

                        # calculate loss
                        constraints = self.constraints(lstm_model.lstm.parameters())
                        if reg_methode == 'log_barrier':
                            flatten_params = self.flatten_params(lstm_model.lstm.parameters())
                            reg, paras = self.backtracking_line_search(flatten_params, 0.1)
                            params = self.write_flat_params(self.temp_lstm, flatten_params)
                            lstm_model.parameters = params
                        else:
                            reg = self.regularization_term(constraints, r, thd)

                        output = lstm_model(batch_cases).to(torch.float32).to(device)
                        labels = labels.to(torch.float32).to(device)
                        loss_ = criterion(output, labels)
                        loss = loss_ + reg

                        """ backpropagation """
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        if loss.item() < self.tol:
                            break_flag = True
                            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, self.max_epochs, loss.item()))
                            print("The loss value is reached")
                            break
                        elif loss_prev is not None and np.abs(np.mean(loss_prev - loss.item()) / np.mean(loss_prev)) < 1e-6:
                            break_flag = True
                            print(np.mean(loss_prev - loss.item()) / np.mean(loss_prev))
                            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, self.max_epochs, loss.item()))
                            print("The loss changes no more")
                            break
                        elif (epoch + 1) % 50 == 0:
                            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, self.max_epochs, loss.item()))
                        loss_prev = loss.item()

                    if break_flag:
                        break

                """ save model """
                self.save_model(lstm_model, r, thd)

    def save_model(self, model, r, thd):
        """ model save path """
        model_save_path = 'models/model_sl_{}_bs_{}_hs_{}_ep_{}_tol_{}_r_{}_thd_{}____.pth'.format(self.seq_len,
                                                                                               self.batch_size,
                                                                                               self.hidden_size,
                                                                                               self.max_epochs,
                                                                                               self.tol, r, thd)

        torch.save(model.state_dict(), model_save_path)


def main():
    trainer = IssLstmTrainer()
    trainer.train_begin(reg_methode='log_barrier')


if __name__ == '__main__':
    main()



