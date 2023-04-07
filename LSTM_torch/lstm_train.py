# -*- coding: utf-8 -*- 
# @Time : 2022/12/22 1:01 
# @Author : Yinan 
# @File : lstm_train.py
import copy
import os.path

import pandas as pd
from networks import LstmRNN, PidNN
import torch
from torch import nn

from utilities import DataCreater, GetLoader, cal_constraints, SaveLoss
from torch.utils.data import DataLoader
import numpy as np

from regularizer import *
from lossfunctions import *
"""
输入： 2 维度，[t-w:t]的y 以及 [t]的u
输出： lstm最后一层隐状态经过Linear层的值
"""

# Define LSTM Neural Networks


class IssLstmTrainer:
    def __init__(self, args):
        self.device = args.device
        self.seq_len = args.len_sequence
        self.input_size = args.input_size
        self.output_size = args.output_size
        self.hidden_size = args.hidden_size
        self.num_layer = args.layers
        self.batch_size = args.batch_size
        self.max_epochs = args.epochs
        self.tol = args.tolerance
        self.tol_stop = args.tol_stop
        self.dataset = args.dataset
        self.curriculum_learning = args.curriculum_learning
        self.reg_methode = args.reg_methode

        self.gamma1 = args.gamma[0]
        self.gamma2 = args.gamma[1]
        self.threshold = args.threshold

        self.lossfcn = None
        self.regularizer = None
        self.dynamic_k = args.dynamic_K
        self.K_pid = args.PID_coefficient
        self.window = args.window
        self.loss_saver = SaveLoss(args.threshold, window=args.window)
        print(self.dynamic_k)

    def train_begin(self):
        device = self.device if torch.cuda.is_available() else 'cpu'
        if device == 'cuda:0':
            print('Training on GPU.')
        else:
            print('No GPU available, training on CPU.')

        #  data set
        data = [r'../data/{}/train/train_input.csv'.format(self.dataset)
                , r'../data/{}/train/train_output.csv'.format(self.dataset)
                , r'../data/{}/val/val_input.csv'.format(self.dataset)
                , r'../data/{}/val/val_output.csv'.format(self.dataset)]
        train_x, train_y, _ = DataCreater(data[0], data[1], data[2], data[3], self.input_size
                                       , self.output_size).creat_new_dataset(seq_len=self.seq_len)
        train_set = GetLoader(train_x, train_y, window_size=self.seq_len, train=True)
        train_set = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=2)

        # ----------------- train -------------------
        lstm_model = LstmRNN(self.input_size + self.output_size, self.hidden_size, output_size=self.output_size
                             , num_layers=self.num_layer)

        Pid_NN = PidNN((self.input_size + self.output_size) * self.batch_size * self.seq_len)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam([{'params': lstm_model.parameters()}, {'params': Pid_NN.parameters()}], lr=1e-3)

        lstm_model.to(device)
        Pid_NN.to(device)
        criterion.to(device)
        print('LSTM model:', lstm_model)
        print('model.parameters:', lstm_model.parameters)
        break_flag = False
        loss_prev = None
        # reg_loss_prev = [None, None]
        # accumulative_reg_loss = [0, 0]
        gamma1 = self.gamma1
        gamma2 = self.gamma2

        weight_save = pd.DataFrame(columns=['norm i', 'sigmoid norm i',  'norm o', 'sigmoid norm o'
            , 'norm f', 'sigmoid norm f', 'norm cell state', 'c1', 'c2', 'reg_loss1', 'reg_loss2', 'loss_'])

        # loss function
        if self.reg_methode == 'log_barrier_BLS':
            self.lossfcn = LossBLS(lstm_model=lstm_model)
        elif self.reg_methode == 'relu':
            self.lossfcn = LossRelu()
        elif self.reg_methode == 'vanilla':
            self.lossfcn = LossVanilla()
        else:
            raise 'undefined regularization method!'

        # curriculum_learning
        if self.curriculum_learning is not None:
            if self.curriculum_learning == 'balance':  # TODO: weight of reg_loss should decrease with decreasing reg_loss
                self.regularizer = BlaRegularizer()

            # 'exp' and '2zero' have similar idea: enforce constraints to a negative value closed to zero
            elif self.curriculum_learning == 'exp':
                self.regularizer = ExpRegularizer()
            elif self.curriculum_learning == '2zero':
                self.regularizer = ToZeroRegularizer()
            elif self.curriculum_learning == 'PID':   # control reg loss to 0 with variable threshold
                self.regularizer = PIDRegularizer(self.K_pid)
            elif self.curriculum_learning == 'IncrePID':
                self.regularizer = Incremental_PIDRegularizer(self.K_pid)
            else:
                raise 'undefined curriculum strategy'

        k_list = pd.DataFrame()
        # results = torch.empty(1, 1).to(device)
        for epoch in range(self.max_epochs):
            print(epoch)
            for batch_cases, labels in train_set:
                batch_cases = batch_cases.transpose(0, 1)
                labels = labels.transpose(0, 1)
                batch_cases = batch_cases.to(torch.float32).to(device)

                # calculate loss
                constraints, weight_save = cal_constraints(self.hidden_size, lstm_model.lstm.parameters(), df=weight_save)
                _, reg_loss = self.lossfcn.forward(constraints, self.threshold)

                overshoot, response, steady_error = self.loss_saver.add_loss(torch.tensor([reg_loss]), epoch)

                output = lstm_model(batch_cases)[-1, :, :].to(torch.float32).to(device)
                labels = labels.to(torch.float32)[0, :, :].to(device)
                loss_ = criterion(output, labels)

                # if epoch == self.max_epochs-1:
                #     results = torch.cat([results, output], dim=0)
                if self.curriculum_learning == 'PID' and self.dynamic_k:
                    dynamic_k = Pid_NN(batch_cases.reshape(-1)).to(torch.float32).to(device)
                    k_list = pd.concat([k_list, pd.DataFrame(dynamic_k.cpu().detach().numpy().reshape(1, -1))], axis=0)
                    self.regularizer = PIDRegularizer(dynamic_k)

                relu_loss_fcn = LossRelu()
                _, relu_loss = relu_loss_fcn(constraints, self.threshold)

                gamma1, gamma2 = self.regularizer.forward(loss_, reg_loss)

                if self.curriculum_learning == 'PID' and self.dynamic_k and None not in [overshoot, response, steady_error]:
                    overshoot = overshoot.float().to(device)
                    response = response.float().to(device)
                    steady_error = steady_error.float().to(device)

                    # print('overshoot:', overshoot)
                    # print('response:', response)
                    # print('steady error', steady_error)

                    reg_k_loss = torch.dot(overshoot, dynamic_k[0, :]) + torch.dot(overshoot, 1 / dynamic_k[1, :]) + \
                           torch.dot(overshoot, 1 / dynamic_k[2, :]) + \
                           torch.dot(response, 1 / dynamic_k[0, :]) + torch.dot(steady_error, dynamic_k[1, :]) + \
                           torch.dot(response, 1 / dynamic_k[2, :])

                    # loss = loss_ + gamma1 * reg_loss[0] + gamma2 * reg_loss[1] + reg_k_loss
                    loss = loss_
                    # print(relu_loss[0], tmp[0], relu_loss[1], tmp[1], relu_loss[0] * tmp[0] + relu_loss[1] * tmp[1])
                    # print(dynamic_k)
                    # print('rl0:{}, tmp0:{}, rl1:{}, tmp1:{}'.format(relu_loss[0], relu_loss[1], tmp[0], tmp[1]))
                    # print(dynamic_k)

                    # print(loss)
                else:
                    loss = loss_ + gamma1 * reg_loss[0] + gamma2 * reg_loss[1]

                if self.dynamic_k:
                    weight_save.iloc[-1, -3:-1] = [gamma1.item() * reg_loss[0].item(), gamma2.item() * reg_loss[1].item()]
                else:
                    weight_save.iloc[-1, -3:-1] = [gamma1 * reg_loss[0].item(),
                                                   gamma2 * reg_loss[1].item()]
                weight_save.iloc[-1, -1] = loss_.item()

                """ backpropagation """
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if abs(loss.item()) < self.tol:
                    break_flag = True
                    print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, self.max_epochs, loss.item()))
                    print("The loss value is reached")
                    break
                elif loss_prev is not None and np.abs(np.mean(loss_prev - loss.item()) / np.mean(loss_prev)) < self.tol_stop:
                    break_flag = True
                    print(np.mean(loss_prev - loss.item()) / np.mean(loss_prev))
                    print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, self.max_epochs, loss.item()))
                    print("The loss changes no more")
                    break
                elif (epoch + 1) % 10 == 0:
                    print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, self.max_epochs, loss.item()))
                loss_prev = loss.item()
                # accumulative_reg_loss[0] += reg_loss[0].item()
                # accumulative_reg_loss[1] += reg_loss[1].item()
                # reg_loss_prev[0], reg_loss_prev[1] = reg_loss[0].item(), reg_loss[1].item()

            if break_flag:
                break

        """ save model """
        self.save_model(self.reg_methode, self.curriculum_learning, lstm_model, [gamma1, gamma2], thd=self.threshold)
        weight_save.to_csv('./statistic/{}/weights_{}_{}.csv'.format(self.dataset, self.reg_methode,
                                                                             self.curriculum_learning), index=False)
        if self.dynamic_k:
            k_list.to_csv('./statistic/{}/K_{}_{}.csv'.format(self.dataset, self.reg_methode,
                                                                             self.curriculum_learning), index=False)
        # pd.DataFrame(results.cpu().detach()).to_csv('./statistic/{}/Test_{}_{}.csv'.format(self.dataset, self.reg_methode,
        #                                                                      self.curriculum_learning), index=False)


    def save_model(self, methode, curriculum_learning, model, gamma, thd):
        """ model save path """
        model_save_path = 'models/{}/curriculum_{}/{}/model_sl_{}_bs_{}_hs_{}_ep_{}_tol_{}_gm_[{:.3g}' \
                          ',{:.3g}]_thd_[{:.3g},{:.3g}].pth'.format(self.dataset, curriculum_learning, methode
                            , self.seq_len, self.batch_size, self.hidden_size, self.max_epochs
                            ,self.tol, gamma[0], gamma[1], thd[0], thd[1])

        if not os.path.exists('models/{}/curriculum_{}/{}'.format(self.dataset, curriculum_learning, methode)):
            os.makedirs('models/{}/curriculum_{}/{}'.format(self.dataset, curriculum_learning, methode))
        torch.save(model.state_dict(), model_save_path)
# TODO: L^* = argmax_{x1, x2} / L(x_1, x_2)
# TODO: https://arxiv.org/abs/1412.6572


def main(args):
    trainer = IssLstmTrainer(args)
    trainer.train_begin()


# if __name__ == '__main__':
#     main(parser.parse_args())




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