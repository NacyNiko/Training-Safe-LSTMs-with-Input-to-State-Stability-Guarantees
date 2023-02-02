# -*- coding: utf-8 -*- 
# @Time : 2022/12/22 1:01 
# @Author : Yinan 
# @File : lstm_train.py
import copy

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
"""


# Define LSTM Neural Networks
class LstmRNN(nn.Module):
    def __init__(self, input_size=2, hidden_size=5, output_size=1, num_layers=1, mean=-10., std=4.):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  #  [seq_len, batch_size, input_size] --> [seq_len, batch_size, hidden_size]
        self.linear1 = nn.Linear(hidden_size, output_size)  #  [seq_len, batch_size, hidden_size] --> [seq_len, batch_size, output_size]

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = x.view(s, b, -1)
        return x[-1, :, :]


class IssLstmTrainer:
    def __init__(self, seq_len=5, input_size=2, hidden_size=5, output_size=1, num_layer=1, batch_size=64,
                 max_epochs=100, tol=1e-5, gamma_list=[(2,2)], threshold=[(1,1)]):
        self.seq_len = seq_len
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layer = num_layer
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.tol = tol
        self.gamma_list = torch.tensor(gamma_list)
        self.threshold = torch.tensor(threshold)

        self.old_model = LstmRNN().to('cuda:0')
        self.old_model.load_state_dict(torch.load(
            './models/curriculum_False/relu/model_sl_5_bs_64_hs_5_ep_100_tol_1e-05_r_tensor([1., 1.])_thd_tensor([0., 0.]).pth'))
        self.temp_lstm = LstmRNN().lstm

    @staticmethod
    def nrmse(y, y_hat):  # normalization to y
        y = y.squeeze(1)
        return np.sqrt(1 / y.shape[0]) * torch.norm(y - y_hat)

    @staticmethod
    def regularization_term(cons, threshold, relu=True):  # paras = model.lstm.parameters()  [W, U, b1, b2]
        con1, con2 = cons[0], cons[1]
        if relu:
            return torch.relu(con1 + threshold[0]) + torch.relu(con2 + threshold[1])
        else:
            return con1 + threshold[0] + con2 + threshold[1]
    @staticmethod
    def log_barrier(cons, t=0.05):
        barrier = 0
        for con in cons:
            if con < 0:
                barrier += - (1/t) * torch.log(-con)
            else:
                return torch.tensor(float('inf'))
        return barrier
    @staticmethod
    def curriculum_learning(cons, grad):
        con1, con2 = cons[0], cons[1]

    def backtracking_line_search(self, new_model, alpha=0.1, typ='normal'):
        cons = self.constraints(new_model.lstm.parameters())
        bar_loss = self.log_barrier(cons)
        ls = 0
        while torch.isinf(bar_loss):
            new_params = self.flatten_params(new_model.lstm.parameters())
            old_params = self.flatten_params(self.old_model.lstm.parameters())
            new_paras = alpha * old_params + (1-alpha) * new_params
            self.temp_lstm = self.write_flat_params(self.temp_lstm, new_paras)
            cons = self.constraints(self.temp_lstm.parameters())
            bar_loss = self.log_barrier(cons)
            ls += 1
            new_model.lstm = self.update_model(self.temp_lstm, new_model.lstm)
            if ls == 500:
                # print('maximum search times reached')
                raise 'maximum search times reached'
        self.old_model.lstm = self.update_model(new_model.lstm, self.old_model.lstm)
        return bar_loss, new_model.lstm.state_dict()

    def update_model(self, model1, model2):
        dict1 = model1.state_dict()
        dict2 = model2.state_dict()
        for par1, par2 in zip(dict1, dict2):
            dict2[par2] = dict1[par1]
        model2.load_state_dict(dict2)
        return model2

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
        return lstm

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

    def train_begin(self, device='cuda:0' if torch.cuda.is_available() else 'cpu'
                    , reg_methode='relu', curriculum_learning=False, gamma=1, threshold=torch.tensor([1., 1.])):
        if device == 'cuda:0':
            print('Training on GPU.')
        else:
            print('No GPU available, training on CPU.')

        # for gamma in self.gamma_list:
        #     for thd in self.threshold:
        """ data set """
        data = [r'../data/train/train_input_noise.csv', r'../data/train/train_output_noise.csv']
        train_x, train_y = DataCreater(data[0], data[1]).creat_new_dataset(seq_len=self.seq_len)
        train_set = GetLoader(train_x, train_y)
        train_set = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=2)
        # ----------------- train -------------------
        lstm_model = LstmRNN(self.input_size, self.hidden_size, output_size=self.output_size, num_layers=self.num_layer)
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
                if reg_methode == 'log_barrier_BLS':
                    _, new_params = self.backtracking_line_search(copy.deepcopy(lstm_model), 0.1)
                    for w in new_params:
                        lstm_model.lstm.state_dict()[w].copy_(new_params[w])
                    constraints = self.constraints(lstm_model.lstm.parameters())
                    reg_loss = self.log_barrier(constraints)
                elif reg_methode == 'relu':
                    reg_loss = self.regularization_term(constraints, threshold=threshold, relu=True)
                else:
                    raise 'undefined regularization method!'
                output = lstm_model(batch_cases).to(torch.float32).to(device)
                labels = labels.to(torch.float32).to(device)
                loss_ = criterion(output, labels)

                if curriculum_learning:
                    if reg_methode == 'relu' and reg_loss != 0:
                        gamma = loss_.detach() / reg_loss.detach()
                        # gamma = torch.exp(reg_loss.detach())
                loss = loss_ + gamma * reg_loss

                """ backpropagation """
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if loss.item() < self.tol:
                    break_flag = True
                    print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, self.max_epochs, loss.item()))
                    print("The loss value is reached")
                    break
                elif loss_prev is not None and np.abs(np.mean(loss_prev - loss.item()) / np.mean(loss_prev)) < 1e-10:
                    break_flag = True
                    print(np.mean(loss_prev - loss.item()) / np.mean(loss_prev))
                    print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, self.max_epochs, loss.item()))
                    print("The loss changes no more")
                    break
                elif (epoch + 1) % 10 == 0:
                    print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, self.max_epochs, loss.item()))
                loss_prev = loss.item()

            if break_flag:
                break

        """ save model """
        self.save_model(reg_methode, curriculum_learning, lstm_model, gamma, thd=threshold)

    def save_model(self, methode, curriculum_learning, model, gamma, thd):
        """ model save path """
        model_save_path = 'models/curriculum_{}/{}/model_sl_{}_bs_{}_hs_{}_ep_{}_tol_{}_r_{}_thd_{}.pth'.format(curriculum_learning
                                                                                               , methode
                                                                                               , self.seq_len,
                                                                                               self.batch_size,
                                                                                               self.hidden_size,
                                                                                               self.max_epochs,
                                                                                               self.tol, gamma, thd)

        torch.save(model.state_dict(), model_save_path)


def main():
    trainer = IssLstmTrainer()
    trainer.train_begin(reg_methode='relu', curriculum_learning=True)


if __name__ == '__main__':
    main()




"""
1. self.old_model 初始化为之前训练的模型，满足log barrier ！= inf
2. 实例化 lstm_model
3. BLS:
    3.1: 验证lstm_model 的Log barrier是否 = inf
        yes：3.1.1: 两个模型参数flatten
             3.1.2: 更新模型
             3.1.3: write_flatten_params
             3.1.4: 验证log barrier
        no: self.old_model = copy.deepcopy(lstm_model) 

"""
"""
Next Step:
visualize the loss of constraints over epochs.
 - compare the differences of different methods.
 - curriculum learning strategy.
 - use different strategy to each constraints.

"""