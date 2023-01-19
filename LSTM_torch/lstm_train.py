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
    def __init__(self, input_size, hidden_size=5, output_size=1, num_layers=1):
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

    @staticmethod
    def nrmse(y, y_hat):  # normalization to y
        y = y.squeeze(1)
        return np.sqrt(1 / y.shape[0]) * torch.norm(y - y_hat)

    def regularization_term(self, paras, r, threshold):  # paras = model.lstm.parameters()  [W, U, b1, b2]
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

        W_f = weight_ih[hidden_size:2*hidden_size, :]
        U_f = weight_hh[hidden_size:2*hidden_size, :]
        b_f = bias[hidden_size:2*hidden_size].unsqueeze(1)

        W_i = weight_ih[:hidden_size, :]
        U_i = weight_hh[:hidden_size, :]
        b_i = bias[:hidden_size].unsqueeze(1)

        U_c = weight_hh[2 * hidden_size: 3 * hidden_size, :]

        con1 = (1 + torch.sigmoid(torch.norm(torch.hstack((W_o, U_o, b_o)), torch.inf))) * \
               torch.sigmoid(torch.norm(torch.hstack((W_f, U_f, b_f)), torch.inf)) - 1
        con2 = (1 + torch.sigmoid(torch.norm(torch.hstack((W_o, U_o, b_o)), torch.inf))) * \
               torch.sigmoid(torch.norm(torch.hstack((W_i, U_i, b_i)), torch.inf)) * torch.norm(U_c, 1) - 1
        return r[0] * torch.relu(con1 + threshold[0]) + r[1] * torch.relu(con2 + threshold[1])
    # return torch.dot(r, torch.relu((torch.tensor([con1, con2]) + threshold)))

    def train_begin(self, device='cuda:0' if torch.cuda.is_available() else 'cpu', ):
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
                        output = lstm_model(batch_cases).to(torch.float32).to(device)
                        labels = labels.to(torch.float32).to(device)

                        # calculate loss
                        reg = self.regularization_term(lstm_model.lstm.parameters(), r, thd)
                        loss_ = criterion(output, labels)

                        # loss__ = loss_ + reg
                        # loss = reg / loss__ * loss_ + loss_ / loss__ * reg
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
    trainer.train_begin()



    # """ hyperparameter I"""
    # train = True
    # seq_len = 5
    # batch_size = 64
    # hidden_size = 5
    # max_epochs = 100
    # INPUT_FEATURES_NUM = 2
    # OUTPUT_FEATURES_NUM = 1
    # tol = 1e-5
    #
    # # checking if GPU is available
    # device = torch.device("cpu")
    # use_gpu = torch.cuda.is_available()
    #
    # if train:
    #     """ hyperparameter II"""
    #     if use_gpu:
    #         device = torch.device("cuda:0")
    #         print('Training on GPU.')
    #     else:
    #         print('No GPU available, training on CPU.')
    #
    #     prev_loss = 1000
    #     # r: weight  threshold
    #     r_set = list(torch.tensor([1., 1.]) * i for i in range(0, 11))
    #     # threshold
    #     threshold_set = list(torch.tensor([1., 1.]) * i for i in range(0, 11))
    #     for r in r_set:
    #         for threshold in threshold_set:
    #
    #             """ model save path """
    #             model_save_path = 'models/model_sl_{}_bs_{}_hs_{}_ep_{}_tol_{}_r_{}_thd_{}.pth'.format(seq_len, batch_size, hidden_size,
    #                                                                                        max_epochs, tol, r, threshold)
    #             """ data set """
    #             data = [r'../data/train/train_input_noise.csv', r'../data/train/train_output_noise.csv']
    #             train_x, train_y = DataCreater(data[0], data[1]).creat_new_dataset(seq_len=seq_len)
    #             train_set = GetLoader(train_x, train_y)
    #             train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
    #
    #             # ----------------- train -------------------
    #             lstm_model = LstmRNN(INPUT_FEATURES_NUM, hidden_size, output_size=OUTPUT_FEATURES_NUM, num_layers=1)
    #             criterion = nn.MSELoss()
    #             optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
    #
    #             lstm_model.to(device)
    #             criterion.to(device)
    #             print('LSTM model:', lstm_model)
    #             print('model.parameters:', lstm_model.parameters)
    #
    #             break_flag = False
    #             for epoch in range(max_epochs):
    #                 for batch_cases, labels in train_set:
    #                     batch_cases = batch_cases.transpose(0, 1)
    #                     batch_cases = batch_cases.transpose(0, 2).to(torch.float32).to(device)
    #                     output = lstm_model(batch_cases).to(torch.float32).to(device)
    #                     labels = labels.to(torch.float32).to(device)
    #
    #                     # calculate loss
    #                     reg = regularization_term(lstm_model.lstm.parameters(), hidden_size, r, threshold)
    #                     loss_ = criterion(output, labels)
    #                     loss = loss_ + reg
    #
    #                     """ backpropagation """
    #                     optimizer.zero_grad()
    #                     loss.backward()
    #                     optimizer.step()
    #
    #                     if loss < prev_loss:
    #                         torch.save(lstm_model.state_dict(), 'lstm_model.pt')  # save model parameters to files
    #                         prev_loss = loss
    #
    #                     if loss.item() < tol:
    #                         break_flag = True
    #                         print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
    #                         print("The loss value is reached")
    #                         break
    #                     elif (epoch + 1) % 10 == 0:
    #                         print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))
    #                 if break_flag:
    #                     break
    #             torch.save(lstm_model.state_dict(), model_save_path)

    # else:
    #     """ eval """
    #     data_t, n_t = [r'../data/train/train_input_noise.csv', r'../data/train/train_output_noise.csv'], 'train'
    #     data_v, n_v = [r'../data/val/val_input_noise.csv', r'../data/val/val_output_noise.csv'], 'val'
    #     net = LstmRNN(2, hidden_size, output_size=1, num_layers=1)
    #
    #     models = os.listdir('./models')
    #
    #     for model in models:
    #         load_path = './models/' + model
    #         net.load_state_dict(torch.load(load_path))
    #
    #         net.eval()
    #         net.to(device)
    #
    #         f, ax = plt.subplots(2, 1)
    #         f.suptitle('Model: ' + model[:-4])
    #         i = 0
    #         for data, n in [[data_t, n_t], [data_v, n_v]]:
    #             data_x, data_y = DataCreater(data[0], data[1]).creat_new_dataset(
    #                 seq_len=seq_len)
    #             data_set = GetLoader(data_x, data_y)
    #
    #             data_set = DataLoader(data_set, batch_size=1, shuffle=False, drop_last=False, num_workers=2)
    #             predictions = list()
    #
    #             with torch.no_grad():
    #                 for batch_case, label in data_set:
    #                     label.to(device)
    #                     batch_case = batch_case.transpose(0, 1)
    #                     batch_case = batch_case.transpose(0, 2).to(torch.float32).to(device)
    #                     predict = net(batch_case).to(torch.float32).to(device)
    #                     predictions.append(predict.squeeze(0).squeeze(0).cpu())
    #
    #             fit_score = nrmse(data_y, torch.tensor(predictions))
    #
    #             ax[i].plot(predictions, color='m', label='pred', alpha=0.8)
    #             ax[i].plot(data_y, color='c', label='real', linestyle='--', alpha=0.5)
    #             ax[i].tick_params(labelsize=5)
    #             ax[i].legend(loc='best')
    #             ax[i].set_title('NRMSE on {} set: {:.3f}'.format(n, fit_score), fontsize=8)
    #             i += 1
    #         plt.savefig('./results/{}.jpg'.format(model[:-4]), bbox_inches='tight', dpi=500)
    #     # plt.show()

if __name__ == '__main__':
    main()



