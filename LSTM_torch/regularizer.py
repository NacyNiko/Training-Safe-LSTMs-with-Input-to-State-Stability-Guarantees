# -*- coding: utf-8 -*- 
# @Time : 2023/2/13 20:35 
# @Author : Yinan 
# @File : regularizer.py

import torch.nn as nn
import torch


class Regularizer(nn.Module):
    def __init__(self):
        super(Regularizer, self).__init__()
        self.reg_loss = None
        self.loss = None

    def forward(self, loss, reg_loss):
        pass


class PIDRegularizer(Regularizer):
    def __init__(self, k):
        super(PIDRegularizer, self).__init__()
        self.prev_reg_loss = [0, 0]
        self.acc_reg_loss = [0, 0]
        self.Kp, self.Ki, self.Kd = k
        self.gamma = [0, 0]

    def forward(self, loss, reg_loss):
        self.reg_loss = reg_loss
        self.loss = loss
        for i in range(2):
            self.gamma[i] = self.Kp[i] * self.reg_loss[i].item() + \
                    self.Ki[i] * self.acc_reg_loss[i] + \
                    self.Kd[i] * (self.reg_loss[i].item() - self.prev_reg_loss[i])
            self.acc_reg_loss[i] += self.reg_loss[i].item()
            self.prev_reg_loss[i] = self.reg_loss[i].item()
        return self.gamma[0], self.gamma[1]


class Incremental_PIDRegularizer(PIDRegularizer):
    def __init__(self, k):
        super(Incremental_PIDRegularizer, self).__init__(k)
        self.pprev_reg_loss = [0, 0]

    def forward(self, loss, reg_loss):
        self.reg_loss = reg_loss
        self.loss = loss
        delta_gamma = [0, 0]
        for i in range(2):
            delta_gamma[i] = self.Kp[i] * (self.reg_loss[i].item() - self.prev_reg_loss[i]) +\
                            self.Ki[i] * self.reg_loss[i].item() - \
                            self.Kd[i] * (self.reg_loss[i].item() - 2 * self.prev_reg_loss[i] + self.pprev_reg_loss[i])
            self.pprev_reg_loss[i] = self.prev_reg_loss[i]
            self.prev_reg_loss[i] = self.reg_loss[i].item()

        self.gamma[0] += delta_gamma[0]
        self.gamma[1] += delta_gamma[1]
        return self.gamma[0], self.gamma[1]


class TwopartRegularizer(Regularizer):
    def __init__(self):
        super(TwopartRegularizer, self).__init__()

    def forward(self, loss, reg_loss):
        self.reg_loss = reg_loss
        self.loss = loss
        temp = []
        for i in range(2):
            if self.reg_loss[i] < 0:
                gamma = 0.01
            else:
                gamma = 1
            temp.append(gamma)
        gamma1, gamma2 = temp
        return gamma1, gamma2


class ToZeroRegularizer(Regularizer):
    def __init__(self):
        super(ToZeroRegularizer, self).__init__()

    def forward(self, loss, reg_loss):
        self.reg_loss = reg_loss
        self.loss = loss
        temp = []
        for i in range(2):
            if self.reg_loss[i] < 0:
                gamma = -0.05
            else:
                gamma = 1
            temp.append(gamma)
        gamma1, gamma2 = temp
        return gamma1, gamma2


class ExpRegularizer(Regularizer):
    def __init__(self):
        super(ExpRegularizer, self).__init__()

    def forward(self, loss, reg_loss):
        self.reg_loss = reg_loss
        self.loss = loss
        temp = []
        for i in range(2):
            if self.reg_loss[i] < 0:
                gamma = -torch.exp(max(self.reg_loss[i].detach(), -torch.tensor(10)))
            else:
                gamma = torch.exp(min(self.reg_loss[0].detach(), torch.tensor(10)))
            temp.append(gamma)
        gamma1, gamma2 = temp
        # gamma1 = torch.exp(min(self.reg_loss[0].detach(), torch.tensor(10)))
        # gamma2 = torch.exp(min(self.reg_loss[1].detach(), torch.tensor(10)))
        return gamma1, gamma2


class BlaRegularizer(Regularizer):
    def __init__(self):
        super(BlaRegularizer, self).__init__()

    def forward(self, loss, reg_loss):
        self.reg_loss = reg_loss
        self.loss = loss
        temp = []
        for i in range(2):
            if self.reg_loss[i].detach() > 0:
                gamma = self.loss.detach() / self.reg_loss[i].detach()
            else:
                gamma = 0
            temp.append(gamma)
        gamma1, gamma2 = temp
        return gamma1, gamma2

