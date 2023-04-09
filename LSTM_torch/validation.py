# -*- coding: utf-8 -*- 
# @Time : 2022/12/13 14:41 
# @Author : Yinan 
# @File : validation.py
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


class Validator:
    def __init__(self, args, device='cpu'):
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

        # self.l_r = np.array(sum(list([i] * len(args.gamma) for i in range(0, len(args.gamma))), []))
        # self.l_thd = np.array(list(range(0, len(args.threshold))) * len(args.threshold))
        self.l_c = []

    def load_data(self):
        data_t = [r'../data/{}/train/train_input.csv'.format(self.dataset)
                       , r'../data/{}/train/train_output.csv'.format(self.dataset)]
        data_v = [r'../data/{}/val/val_input.csv'.format(self.dataset)
                       , r'../data/{}/val/val_output.csv'.format(self.dataset)]
        return data_t, data_v

    @staticmethod
    def nrmse(y, y_hat):  # normalization to y
        # y = y.squeeze(1)
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

    def evaluate(self, model, path, data_t, data_v, save_plot=False):
        # _, gamma, thd = name.split('tensor')
        c1, c2 = self.evaluate_constraint(model)

        self.l_c.append((float(c1), float(c2)))

        if save_plot:
            if self.output_size > 1:
                for n in [True, False]:
                    f, ax = plt.subplots(self.output_size, 1, figsize=(30, 10) if n else (10, 10))

                    data_x, data_y = DataCreater(data_t[0], data_t[1], data_v[0], data_v[1]
                                                 , self.input_size, self.output_size, train=n).creat_new_dataset(
                        seq_len=self.seq_len)
                    data_set = GetLoader(data_x, data_y, window_size=self.seq_len, train=False)

                    data_set = DataLoader(data_set, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
                    # predictions = []
                    with torch.no_grad():
                        for batch_case, label in data_set:

                        # for i, batch in enumerate(data_set):
                    #         last_data = batch.squeeze(0)
                    #         init_ph_value = torch.zeros(last_data.shape[0], last_data.shape[1], 1)  # 初始化缺失的PH值
                    #         last_data = torch.cat([last_data, init_ph_value], dim=1)  # 将流速信息与初始化的PH值拼接
                    #
                    #         with torch.no_grad():
                    #             output = model(last_data)[:, -1].unsqueeze(1)  # 使用先前预测的PH值作为输入
                    #             predictions.append(output)
                    #
                    #         if i < len(data_set) - 1:
                    #             next_data = data_set.dataset[i + 1]
                    #             next_data[:, :, 1] = torch.cat([next_data[:, 1:, 1], output], dim=1)  # 更新下一个输入的PH值
                    #
                    #     predictions = torch.cat(predictions, dim=1)

                            label.to(self.device)
                            batch_case = batch_case.transpose(0, 1)
                            batch_case = batch_case.to(torch.float32).to(self.device)
                            predict = model(batch_case).to(torch.float32).to(self.device)
                            predictions = torch.concat([predictions, predict.cpu()], dim=0)

                    i = 0
                    while i < self.output_size:
                        fit_score = self.nrmse(data_y[:, i], predictions[1:, i].clone().detach())
                        f.suptitle('Model: ' + path[18:-4] + 'c1:{} c2:{} {}'.format(c1, c2, self.dynamic_K))
                        ax[i].plot(predictions[1:, i], color='m', label='pred', alpha=0.8)
                        ax[i].plot(data_y[:, i], color='c', label='real', linestyle='--', alpha=0.5)
                        ax[i].tick_params(labelsize=5)
                        ax[i].legend(loc='best')
                        ax[i].set_title('NRMSE on {} set: {:.3f}'.format(n, fit_score), fontsize=8)
                        i += 1
                    plt.savefig('./results{}{}.jpg'.format(path[6:-4], 'train' if n else 'val'), bbox_inches='tight', dpi=500)
            else:
                f, ax = plt.subplots(2, 1)
                j = 0
                for n in [True, False]:
                    data_x, data_y = DataCreater(data_t[0], data_t[1], data_v[0], data_v[1],
                                                 self.input_size, self.output_size, train=n).creat_new_dataset(
                        seq_len=self.seq_len)
                    data_set = GetLoader(data_x, data_y, window_size=self.seq_len, train=False)

                    data_set = DataLoader(data_set, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
                    # predictions = torch.empty(1, 1, self.output_size)
                    predictions = []
                    with torch.no_grad():
                        i = 0
                        for batch, label in data_set:
                            batch = batch.transpose(0, 1)
                            batch = batch.to(torch.float32).to(self.device)
                            if i == 0:
                                batch = batch
                                pre_prediction = batch[:, :, :self.output_size]                # 将流速信息与初始化的PH值拼接
                            else:
                                batch = torch.cat([pre_prediction, batch[:, :, -self.input_size].unsqueeze(1)], dim=2)

                            with torch.no_grad():
                                output = model(batch)  # 使用先前预测的PH值作为输入
                                predictions.append(output[0, :, :])

                            if i < len(data_set) - 1:
                                pre_prediction[:-1, :, :] = pre_prediction[1:, :, :]
                                pre_prediction[-2, :, :] = output[0, :, :]
                            i += 1
                        predictions = torch.tensor(predictions)

                    #     for batch_case, label in data_set:
                    #         label.to(self.device)
                    #         batch_case = batch_case.transpose(0, 1)
                    #         batch_case = batch_case.to(torch.float32).to(self.device)
                    #         predict = model(batch_case).to(torch.float32).to(self.device)
                    #         predictions = torch.concat([predictions, predict[-1, :, :].unsqueeze(1).cpu()], dim=0)
                    # predictions = predictions[1:, :, :].squeeze()

                    fit_score = self.nrmse(data_y[0, :, :], predictions.clone().detach())
                    f.suptitle('Model: ' + path[18:-4] + 'c1:{} c2:{} {}'.format(c1, c2, self.dynamic_K))
                    ax[j].plot(data_y[0, :, :], color='c', label='real', linestyle='--', alpha=0.5)
                    ax[j].plot(predictions, color='m', label='pred', alpha=0.8)
                    ax[j].tick_params(labelsize=5)
                    ax[j].legend(loc='best')
                    ax[j].set_title('NRMSE on {} set: {:.3f}'.format('train' if n else 'val', float(fit_score)))
                    j += 1

                plt.savefig('./results{}{}.jpg'.format(path[6:-4], 'train' if n else 'val'), bbox_inches='tight', dpi=500)

    def evaluate_constraint(self, model):
        parameters = model.lstm.parameters()
        c, _ = cal_constraints(self.hidden_size, parameters)
        return c[0], c[1]


def main(args, if_filter=True, plt3D=False):   # if_filter: ignore whether gamma=0 or threshold=0
    validator = Validator(args, device='cuda')
    # validator = Validator([*range(11)], [*range(11)], device='cuda')
    data_train, data_val = validator.load_data()
    lstmmodel = validator.create_model()
    file = 'models/{}/curriculum_{}/{}/'.format(args.dataset, args.curriculum_learning, args.reg_methode)
    if not os.path.exists('results/{}/curriculum_{}/{}/'.format(args.dataset, args.curriculum_learning, args.reg_methode)):
        os.makedirs('results/{}/curriculum_{}/{}/'.format(args.dataset, args.curriculum_learning, args.reg_methode))
    models = os.listdir(file)
    save_jpgs = os.listdir('results/{}/curriculum_{}/{}/'.format(args.dataset, args.curriculum_learning
                                                                 , args.reg_methode))
    # models = ['model_sl_5_bs_64_hs_5_ep_500_tol_1e-05_r_tensor([2, 2])_thd_tensor([1, 1]).pth']
    for model in models:
        temp1 = model[:-4] + 'val.jpg'
        temp2 = model[:-4] + 'train.jpg'
        if not (temp1 in save_jpgs or temp2 in save_jpgs):
            path = file + model
            lstmmodel = validator.load_model(lstmmodel, path)
            validator.evaluate(lstmmodel, path, data_train, data_val, save_plot=True)

    # if if_filter:
    #     idx = validator.l_r * validator.l_thd
    #     gamma = validator.l_r[idx != 0]
    #     thd = validator.l_thd[idx != 0]
    #     c = pd.DataFrame(validator.l_c)[idx != 0]
    # else:
    #     gamma = validator.l_r
    #     thd = validator.l_thd
    #     c = pd.DataFrame(validator.l_c)

    # if plt3D:
    #     ax_ = plt.axes(projection='3d')
    #     ax_.scatter3D(gamma, thd, c.iloc[:, 0], c=c.iloc[:, 0], s=500*normalize(c.iloc[:, 1]) if if_filter else 100)  # min: -0.9983   max:-0.9955
    #                                                                                         # smaller dot: more negative
    #
    #     ax_.set_xlabel('ratio')
    #     ax_.set_ylabel('threshold')
    #     ax_.set_zlabel('c1')
    # else:
    #     c.reset_index(drop=True, inplace=True)
    #     df = pd.concat([pd.Series(gamma), pd.Series(thd), c], axis=1)
    #     df.columns = ['r', 'thd', 'c1', 'c2']
    #     df = df.groupby('thd')
    #     fig, ax = plt.subplots(2, 1, sharex=True)
    #     for i, dg in df:
    #         ax[0].plot(dg.loc[:, 'r'], dg.loc[:, 'c1'], label=i)
    #         ax[1].plot(dg.loc[:, 'r'], dg.loc[:, 'c2'], label=i)
    #     ax[0].set_title('constraint 1')
    #     ax[1].set_title('constraint 2')
    #     lines, labels = fig.axes[-1].get_legend_handles_labels()
    #     fig.legend(lines, labels, bbox_to_anchor=(0.74, 0.96), ncol=4, framealpha=1)
    # plt.show()
    print('-------------Finish---------------------')


# if __name__ == '__main__':
#     main(dataset, plt3D=False)
