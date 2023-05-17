# -*- coding: utf-8 -*- 
# @Time : 2023/2/13 23:00 
# @Author : Yinan 
# @File : lossfunctions.py
import torch
import torch.nn as nn
import copy
from utilities import cal_constraints


class LossFcn(nn.Module):
    def __init__(self):
        super(LossFcn, self).__init__()
        self.cons = None
        self.threshold = None


    def forward(self, cons, threshold):
        pass


class LossRelu(LossFcn):
    def __init__(self):
        super(LossRelu, self).__init__()

    def forward(self, cons, threshold):
        self.cons = cons
        self.threshold = threshold
        con1, con2 = self.cons[0], self.cons[1]
        return None, (torch.relu(con1 + self.threshold[0]), torch.relu(con2 + self.threshold[1]))


class LossVanilla(LossFcn):
    def __init__(self, curriculum):
        super(LossVanilla, self).__init__()
        self.curriculum = curriculum

    def forward(self, cons, threshold):
        self.cons = cons
        self.threshold = threshold
        con1, con2 = self.cons[0], self.cons[1]
        # if self.curriculum == 'PID':
        return None, (-con1 + self.threshold[0], -con2 + self.threshold[1])
        # else:
        #     return None, (con1 + self.threshold[0], con2 + self.threshold[1])



class LossBLS(LossFcn):
    def __init__(self, lstm_model=None):
        super(LossBLS, self).__init__()
        self.lstm_model = lstm_model
        self.old_model = copy.deepcopy(lstm_model).to('cuda:0')
        # self.old_model.load_state_dict(torch.load(
        #     'models/pHdata/curriculum_None/relu/model_sl_10_bs_64_hs_5_ep_100_tol_-0.001_gm_[1,1]_thd_[1,1].pth'))
        self.old_model.load_state_dict(torch.load(
            'models/robot_forward/curriculum_PID/vanilla/model_sl_40_bs_128_hs_150_ep_100_tol_-0.001_gm_[0.00152,0.0522]_thd_[-0.05,-0.1].pth'))
        self.temp_lstm = copy.deepcopy(lstm_model.lstm)
        self.hidden_size = lstm_model.lstm.hidden_size

    def forward(self, cons, threshold):
        _, new_params = self.backtracking_line_search(copy.deepcopy(self.lstm_model), threshold, 0.1)
        for w in new_params:
            self.lstm_model.lstm.state_dict()[w].copy_(new_params[w])
        constraints, _ = cal_constraints(self.hidden_size, self.lstm_model.lstm.parameters())
        return self.log_barrier(constraints, threshold)

    def log_barrier(self, cons, thd,t=0.05):
        barrier = 0
        barrier_l = []
        for con, thd in zip(cons, thd):
            if con + thd < 0:
                barrier += - (1 / t) * torch.log(-con - thd)
                barrier_l.append(- (1 / t) * torch.log(-con - thd))
            else:
                return torch.tensor(float('inf')), []
        return barrier, barrier_l

    def backtracking_line_search(self, new_model, threshold, alpha=0.1):
        cons, _ = cal_constraints(self.hidden_size, new_model.lstm.parameters())
        bar_loss, bar_loss_l = self.log_barrier(cons, thd=threshold)
        ls = 0
        while torch.isinf(bar_loss):
            new_params = self.flatten_params(new_model.lstm.parameters())
            old_params = self.flatten_params(self.old_model.lstm.parameters())
            new_paras = alpha * old_params + (1 - alpha) * new_params
            self.temp_lstm = self.write_flat_params(self.temp_lstm, new_paras)
            cons, _ = cal_constraints(self.hidden_size, self.temp_lstm.parameters())
            bar_loss, _ = self.log_barrier(cons, threshold)
            ls += 1
            new_model.lstm = self.update_model(self.temp_lstm, new_model.lstm)
            if ls == 500:
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
        bias1 = f_params[140:160].resize(20, )
        bias2 = f_params[160:].resize(20, )
        p_list = [W_ih, W_hh, bias1, bias2]
        i = 0
        for p in lstm.parameters():
            p.data = p_list[i]
            i += 1
        return lstm

