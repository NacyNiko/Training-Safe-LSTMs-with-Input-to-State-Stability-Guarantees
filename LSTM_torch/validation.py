# -*- coding: utf-8 -*- 
# @Time : 2022/12/13 14:41 
# @Author : Yinan 
# @File : validation.py
import pickle

import torch
import numpy as np
import pandas as pd
from torch import nn
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from utilities import DataCreater, GetLoader, cal_constraints
from torch.utils.data import DataLoader
from networks import LstmRNN
from sklearn.metrics import r2_score


class Validator:
    def __init__(self, args, device='cpu', if_recoder=False):
        self.dataset = args.dataset
        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.output_size = args.output_size
        self.num_layers = args.layers
        self.seq_len = args.len_sequence
        self.device = device
        self.cur = args.curriculum_learning
        self.reg_mth = args.reg_methode
        self.dynamic_K = args.dynamic_K
        self.gamma = args.gamma[0]
        self.thd = args.threshold[0]
        self.if_record = if_recoder

        if if_recoder:
            if os.path.exists(r'./statistic/{}/record.csv'.format(self.dataset)):
                self.record = pd.read_csv(r'./statistic/{}/record.csv'.format(self.dataset))

            else:
                self.record = pd.DataFrame({'gamma': [0], 'thd': [0], 'NRMSE': [0], 'r2': [0], 'c1': [0], 'c2': [0]})

    def load_data(self):
        data_t = [r'../data/{}/train/train_input.csv'.format(self.dataset)
                       , r'../data/{}/train/train_output.csv'.format(self.dataset)]
        data_v = [r'../data/{}/val/val_input.csv'.format(self.dataset)
                       , r'../data/{}/val/val_output.csv'.format(self.dataset)]
        # data_v = [r'../data/{}/test/test_input.csv'.format(self.dataset)
        #                , r'../data/{}/test/test_output.csv'.format(self.dataset)]
        return data_t, data_v

    @staticmethod
    def nrmse(y, y_hat):
        std = torch.std(y_hat, dim=0)
        norm = torch.norm(y.squeeze() - y_hat.squeeze())
        res = 1 / std * np.sqrt(1 / y.shape[0]) * norm
        return res

    def create_model(self):
        model = LstmRNN(self.input_size + self.output_size, self.hidden_size, self.output_size, self.num_layers)
        return model

    def load_model(self, model, path):
        model.load_state_dict(torch.load(path, map_location=self.device))
        model.eval()
        model.to(self.device)
        return model

    def evaluate(self, model, path, data_t, data_v, z, save_plot=False):
        c1, c2 = self.evaluate_constraint(model)
        # g, t = path.split('_')[-3][-2], path.split('_')[-1][-6]
        # if len(path.split('_')[-3]) > 5:
        #     g = path.split('_')[-3][-3:-1]
        # if len(path.split('_')[-1]) > 9:
        #     t = path.split('_')[-1][-7:-5]
        #
        # dic = {'gamma': [float(g)], 'thd': [float(t)], 'c1': [float(c1)], 'c2': [float(c2)]}
        # temp = pd.DataFrame(dic)
        # self.record = pd.concat([self.record, temp], axis=0)
        # self.record.to_csv(r'./statistic/{}/record.csv'.format(self.dataset))

        if save_plot:
            if self.output_size > 1:
                for n in [True, False]:
                    hidden = (torch.zeros([self.num_layers, 1, self.hidden_size]).to(self.device)
                              , torch.zeros([self.num_layers, 1, self.hidden_size]).to(self.device))
                    plt.close()
                    f, ax = plt.subplots(self.output_size, 1, figsize=(30, 10) if n else (10, 10))

                    data_x, data_y, stat_x, stat_y = DataCreater(data_t[0], data_t[1], data_v[0], data_v[1],
                                                 self.input_size, self.output_size, train=n).creat_new_dataset(
                        seq_len=self.seq_len)
                    stat_x[0] = stat_x[0].to(self.device)
                    stat_x[1] = stat_x[1].to(self.device)
                    stat_y[0] = stat_y[0].to(self.device)
                    stat_y[1] = stat_y[1].to(self.device)
                    data_set = GetLoader(data_x, data_y, seq_len=self.seq_len, train=False)

                    data_set = DataLoader(data_set, batch_size=1, shuffle=False, drop_last=True, num_workers=0)
                    predictions = torch.empty([1, self.output_size]).to(self.device)
                    with torch.no_grad():
                        j = 0
                        for batch, label in data_set:
                            batch = batch.transpose(0, 1)
                            batch = batch.to(torch.float32).to(self.device)

                            if j <= z * self.seq_len:
                                previous_y = batch[:, :, :self.output_size]

                            else:
                                diff = (batch[:, :, :self.output_size] - previous_y) / batch[:, :, :self.output_size]
                                batch[:, :, :self.output_size] = previous_y

                            with torch.no_grad():
                                output, hidden = model(batch)
                                temp = output * stat_y[1] + stat_y[0]
                                predictions = torch.cat([predictions, output * stat_y[1] + stat_y[0]], dim=0)
                                if j >= z * self.seq_len:
                                    previous_y = torch.cat([previous_y, output.unsqueeze(0)], dim=0)
                                    previous_y = previous_y[1:, :, :]
                            j += 1

                    i = 0
                    predictions = predictions.cpu()
                    while i < self.output_size: # z * self.seq_len + 2000
                        fit_score = self.nrmse(data_y[z * self.seq_len:, i]
                                               , predictions[1 + z * self.seq_len:, i].clone().detach())
                        f.suptitle('Model: ' + path[18:-4] + 'c1:{} c2:{} {}'.format(c1, c2, self.dynamic_K))
                        ax[i].plot(predictions[1 + z * self.seq_len:, i], color='m', label='pred', alpha=0.8)
                        ax[i].plot(data_y[z * self.seq_len:, i], color='c', label='real', linestyle='--', alpha=0.5)
                        ax[i].tick_params(labelsize=5)
                        ax[i].legend(loc='best')
                        ax[i].set_title('NRMSE on {} set: {:.3f}'.format(n, fit_score), fontsize=8)
                        i += 1
                    plt.savefig('./results{}{}.jpg'.format(path[6:-4], 'train' if n else 'val'), bbox_inches='tight', dpi=500)
            else:
                plt.close()
                fig, ax = plt.subplots(2, 1)
                j = 0
                for n in [True, False]:
                    hidden = (torch.zeros([self.num_layers, 1, self.hidden_size]).to(self.device)
                              , torch.zeros([self.num_layers, 1, self.hidden_size]).to(self.device))

                    data_x, data_y, stat_x, stat_y = DataCreater(data_t[0], data_t[1], data_v[0], data_v[1],
                                                 self.input_size, self.output_size, train=n).creat_new_dataset(
                        seq_len=self.seq_len)
                    stat_x[0] = stat_x[0].to(self.device)
                    stat_x[1] = stat_x[1].to(self.device)
                    stat_y[0] = stat_y[0].to(self.device)
                    stat_y[1] = stat_y[1].to(self.device)
                    data_set = GetLoader(data_x, data_y, seq_len=self.seq_len, train=False)

                    data_set = DataLoader(data_set, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
                    predictions = []
                    with torch.no_grad():
                        i = 0
                        for batch, label in data_set:
                            batch = batch.transpose(0, 1)
                            batch = batch.to(torch.float32).to(self.device)

                            if i <= z * self.seq_len:
                                previous_y = batch[:, :, :self.output_size]

                            else:
                                batch[:, :, :self.output_size] = previous_y

                            with torch.no_grad():
                                output, hidden = model(batch)
                                predictions.append(output * stat_y[1] + stat_y[0])
                                if i >= z * self.seq_len:
                                    previous_y = torch.cat([previous_y, output.unsqueeze(0)], dim=0)
                                    previous_y = previous_y[1:, :, :]
                            i += 1
                        predictions = torch.tensor(predictions)

                    fit_score = self.nrmse(data_y[z * self.seq_len:], predictions[z * self.seq_len:].clone().detach())

                    fig.suptitle('Model: ' + path[18:-4] + 'c1:{} c2:{} {}'.format(c1, c2, self.dynamic_K))
                    ax[j].plot(data_y[z * self.seq_len:], color='c', label='real', linestyle='--', alpha=0.5)
                    ax[j].plot(predictions[z * self.seq_len:], color='m', label='pred', alpha=0.8)
                    ax[j].tick_params(labelsize=5)
                    ax[j].legend(loc='best')
                    ax[j].set_title('NRMSE on {} set: {:.3f}'.format('train' if n else 'val', float(fit_score)))
                    j += 1

                plt.savefig('./results{}{}.jpg'.format(path[6:-4], 'train' if n else 'val'), bbox_inches='tight', dpi=500)


    def evaluate_piecewise(self, model, path, data_t, data_v, z ,save_plot=False, horizon_window=1):
        c1, c2 = self.evaluate_constraint(model)
        pre_batches = []
        if save_plot:
            if self.output_size > 1:
                for n in [True, False]:
                    fit_score_t, r2_t = 0, 0
                    hidden = (torch.zeros([self.num_layers, 1, self.hidden_size]).to(self.device)
                              , torch.zeros([self.num_layers, 1, self.hidden_size]).to(self.device))
                    plt.close()
                    f, ax = plt.subplots(self.output_size, 1, figsize=(30, 10) if n else (10, 10))

                    data_x, data_y, stat_x, stat_y = DataCreater(data_t[0], data_t[1], data_v[0], data_v[1],
                                                                 self.input_size, self.output_size,
                                                                 train=n).creat_new_dataset(
                        seq_len=self.seq_len)
                    stat_x[0] = stat_x[0].to(self.device)
                    stat_x[1] = stat_x[1].to(self.device)
                    stat_y[0] = stat_y[0].to(self.device)
                    stat_y[1] = stat_y[1].to(self.device)
                    data_set = GetLoader(data_x, data_y, seq_len=self.seq_len, train=False)

                    data_set = DataLoader(data_set, batch_size=1, shuffle=False, drop_last=True, num_workers=0)
                    predictions = torch.empty([1, self.output_size]).to(self.device)
                    with torch.no_grad():
                        j = 0
                        for batch, label in data_set:
                            batch = batch.transpose(0, 1)
                            batch = batch.to(torch.float32).to(self.device)

                            if horizon_window == 1:
                                # warm up
                                if j <= z * self.seq_len:
                                    previous_y = batch[:, :, :self.output_size]
                                with torch.no_grad():
                                    output, hidden = model(batch)
                                    temp = output * stat_y[1] + stat_y[0]
                                    predictions = torch.cat([predictions, output * stat_y[1] + stat_y[0]], dim=0)
                                    if j >= z * self.seq_len:
                                        previous_y = torch.cat([previous_y, output.unsqueeze(0)], dim=0)
                                        previous_y = previous_y[1:, :, :]
                            else:
                                if len(pre_batches) < z * self.seq_len:
                                    pre_batches.append((batch, label))
                                else:
                                    pre_batches.append((batch, label))
                                    pre_batches.pop(0)

                                # initial warm up
                                if j <= z * self.seq_len:
                                    previous_y = batch[:, :, :self.output_size]
                                    with torch.no_grad():
                                        _, _ = model(batch)

                                if j % horizon_window == 0 and j > z * self.seq_len:
                                    # warm up
                                    for w_batch, w_label in pre_batches:
                                        previous_y = w_batch[:, :, :self.output_size]
                                        with torch.no_grad():
                                            output, hidden = model(w_batch)

                                # rebuild batch
                                batch[:, :, :self.output_size] = previous_y
                                output, hidden = model(batch)
                                temp = output * stat_y[1] + stat_y[0]
                                predictions = torch.cat([predictions, output * stat_y[1] + stat_y[0]], dim=0)
                                previous_y = torch.cat([previous_y, output.unsqueeze(0)], dim=0)
                                previous_y = previous_y[1:, :, :]
                                j += 1

                    i = 0
                    predictions = predictions.cpu()
                    while i < self.output_size:  # z * self.seq_len + 2000
                        fit_score = self.nrmse(data_y[z * self.seq_len:, i]
                                               , predictions[1 + z * self.seq_len:, i].clone().detach())
                        r2 = r2_score(data_y[z * self.seq_len:, i]
                                      , predictions[1 + z * self.seq_len:, i].clone().detach())
                        fit_score_t += fit_score
                        r2_t += r2

                        ax[i].plot(predictions[1 + z * self.seq_len:, i], color='m', label='pred', alpha=0.8)
                        ax[i].plot(data_y[z * self.seq_len:, i], color='c', label='real', linestyle='--', alpha=0.5)
                        ax[i].tick_params(labelsize=5)
                        ax[i].legend(loc='best')
                        ax[i].set_title('NRMSE on {} set: {:.3f}, R2: {}'.format(n, fit_score, r2), fontsize=8)
                        i += 1
                    f.suptitle('Model: ' + path[18:-4] + 'NRMSE: {}, r2: {}, c1:{}, c2:{} {}'.format(fit_score_t/6, r2_t/6
                    , c1, c2, self.dynamic_K))
                    plt.savefig('./results{}_{}_{}.jpg'.format(path[6:-4], horizon_window, 'train' if n else 'val'),
                                bbox_inches='tight',
                                dpi=500)
            else:
                plt.close()
                fig, ax = plt.subplots(2, 1)
                j = 0
                for n in [True, False]:
                    data_x, data_y, stat_x, stat_y = DataCreater(data_t[0], data_t[1], data_v[0], data_v[1],
                                                                 self.input_size, self.output_size,
                                                                 train=n).creat_new_dataset(seq_len=self.seq_len)
                    stat_x[0] = stat_x[0].to(self.device)
                    stat_x[1] = stat_x[1].to(self.device)
                    stat_y[0] = stat_y[0].to(self.device)
                    stat_y[1] = stat_y[1].to(self.device)
                    data_set = GetLoader(data_x, data_y, seq_len=self.seq_len, train=False)

                    data_set = DataLoader(data_set, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
                    predictions = []

                    with torch.no_grad():
                        i = 0
                        for batch, label in data_set:
                            batch = batch.transpose(0, 1)
                            batch = batch.to(torch.float32).to(self.device)

                            if horizon_window == 1:
                                # warm up
                                if i <= z * self.seq_len:
                                    previous_y = batch[:, :, :self.output_size]
                                with torch.no_grad():
                                    output, hidden = model(batch)
                                    temp = output * stat_y[1] + stat_y[0]
                                    predictions.append(output * stat_y[1] + stat_y[0])
                                    if i >= z * self.seq_len:
                                        previous_y = torch.cat([previous_y, output.unsqueeze(0)], dim=0)
                                        previous_y = previous_y[1:, :, :]
                            else:
                                if len(pre_batches) < z * self.seq_len:
                                    pre_batches.append((batch, label))
                                else:
                                    pre_batches.append((batch, label))
                                    pre_batches.pop(0)

                                # initial warm up
                                if i <= z * self.seq_len:
                                    previous_y = batch[:, :, :self.output_size]
                                    with torch.no_grad():
                                        _, _ = model(batch)

                                if i % horizon_window == 0 and i > z * self.seq_len:
                                    # warm up
                                    for w_batch, w_label in pre_batches:
                                        previous_y = w_batch[:, :, :self.output_size]
                                        with torch.no_grad():
                                            output, hidden = model(w_batch)

                                # rebuild batch
                                batch[:, :, :self.output_size] = previous_y
                                output, hidden = model(batch)
                                predictions.append(output * stat_y[1] + stat_y[0])
                                previous_y = torch.cat([previous_y, output.unsqueeze(0)], dim=0)
                                previous_y = previous_y[1:, :, :]
                            i += 1
                        predictions = torch.tensor(predictions)

                    fit_score = self.nrmse(data_y[z * self.seq_len:], predictions[z * self.seq_len:].clone().detach())
                    r2 = r2_score(data_y[z * self.seq_len:], predictions[z * self.seq_len:].clone().detach())
                    fig.suptitle('Model: ' + path[18:-4] + 'c1:{} c2:{} {}'.format(c1, c2, self.dynamic_K))
                    ax[j].plot(data_y[z * self.seq_len:], color='c', label='real', linestyle='--', alpha=0.5)
                    ax[j].plot(predictions[z * self.seq_len:], color='m', label='pred', alpha=0.8)
                    ax[j].tick_params(labelsize=5)
                    ax[j].legend(loc='best')
                    ax[j].set_title('NRMSE on {} set: {:.3f}, R2: {}'.format('train' if n else 'val', float(fit_score), float(r2)))
                    j += 1

                    if j == 2 and z == 50:
                        fig_, ax_ = plt.subplots(1, 1)
                        ax_.plot(data_y[:z * self.seq_len+horizon_window], color='c', label='real', linestyle='--', alpha=0.5)
                        ax_.plot(range(z * self.seq_len, z * self.seq_len+horizon_window), predictions[z * self.seq_len:z * self.seq_len+horizon_window], color='m', label='pred', alpha=0.8)
                        ax_.tick_params(labelsize=5)
                        ax_.legend(loc='best')
                        ax_.set_title(
                            'Prediction result on {} set with h={}'.format(self.dataset, horizon_window ))
                        ax_.set_xlabel('Time', fontsize=15)
                        ax_.set_ylabel('pH', fontsize=15)
                        plt.savefig('./results{}_{}_{}.jpg'.format(path[6:-4], horizon_window, 'prediction'),
                                    bbox_inches='tight',
                                    dpi=500)

                plt.savefig('./results{}_{}_{}.jpg'.format(path[6:-4], horizon_window, 'train' if n else 'val'),
                            bbox_inches='tight',
                            dpi=500)

                if self.if_record:
                    self.record_gamma_tau(path, nrmse=fit_score, r2=r2, c1=c1, c2=c2)

    def evaluate_constraint(self, model):
        parameters = model.lstm.parameters()
        c, _ = cal_constraints(self.hidden_size, parameters)
        return c[0], c[1]

    def record_gamma_tau(self, path, nrmse, r2, c1, c2):
        # record
        g, t = path.split('_')[-3][-2], path.split('_')[-1][-6]
        if len(path.split('_')[-3]) > 5:
            g = path.split('_')[-3][-3:-1]
        if len(path.split('_')[-1]) > 9:
            t = path.split('_')[-1][-7:-5]

        dic = {'gamma': [float(g)], 'thd': [float(t)], 'NRMSE': [float(nrmse)], 'r2': [float(r2)],'c1': [float(c1)], 'c2': [float(c2)]}
        temp = pd.DataFrame(dic)
        self.record = pd.concat([self.record, temp], axis=0)
        self.record.to_csv(r'./statistic/{}/record.csv'.format(self.dataset), index=False)


def main(args, if_recoder, piecewise=False):
    validator = Validator(args, device='cuda', if_recoder=if_recoder)
    data_train, data_val = validator.load_data()
    lstmmodel = validator.create_model()
    file = 'models/{}/curriculum_{}/{}/'.format(args.dataset, args.curriculum_learning, args.reg_methode)
    if not os.path.exists('results/{}/curriculum_{}/{}/'.format(args.dataset, args.curriculum_learning, args.reg_methode)):
        os.makedirs('results/{}/curriculum_{}/{}/'.format(args.dataset, args.curriculum_learning, args.reg_methode))
    models = os.listdir(file)
    save_jpgs = os.listdir('results/{}/curriculum_{}/{}/'.format(args.dataset, args.curriculum_learning
                                                                 , args.reg_methode))

    for model in models:
        for hw in [60]:
            temp1 = model[:-4] + f'_{hw}_val.jpg'
            temp2 = model[:-4] + f'_{hw}_train.jpg'
            if not (temp1 in save_jpgs or temp2 in save_jpgs):
                path = file + model
                lstmmodel = validator.load_model(lstmmodel, path)
                if not piecewise:
                    validator.evaluate(lstmmodel, path, data_train, data_val, z=2, save_plot=True)
                else:
                    validator.evaluate_piecewise(lstmmodel, path, data_train, data_val, z=2, save_plot=True, horizon_window=hw)
                    # if hw == 60:
                    #     validator.evaluate_piecewise(lstmmodel, path, data_train, data_val, z=50,save_plot=True, horizon_window=hw)

    print('-------------Finish---------------------')

