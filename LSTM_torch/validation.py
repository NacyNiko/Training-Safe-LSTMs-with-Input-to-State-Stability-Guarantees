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

        # self.l_r = np.array(sum(list([i] * len(args.gamma) for i in range(0, len(args.gamma))), []))
        # self.l_thd = np.array(list(range(0, len(args.threshold))) * len(args.threshold))
        self.l_c = []

    def load_data(self):
        data_t, n_t = [r'../data/{}/train/train_input.csv'.format(self.dataset),
                       r'../data/{}/train/train_output.csv'.format(self.dataset)], 'train'
        data_v, n_v = [r'../data/{}/val/val_input.csv'.format(self.dataset),
                       r'../data/{}/val/val_output.csv'.format(self.dataset)], 'val'
        return [data_t, n_t], [data_v, n_v]

    @staticmethod
    def nrmse(y, y_hat):  # normalization to y
        # y = y.squeeze(1)
        return np.sqrt(1 / y.shape[0]) * torch.norm(y - y_hat)

    def create_model(self):
        model = LstmRNN(self.input_size + self.output_size, self.hidden_size, self.output_size, self.num_layers)
        return model

    def load_model(self, model, path):
        model.load_state_dict(torch.load(path))
        model.eval()
        model.to(self.device)
        return model

    def evaluate(self, model, path, data_t, data_v, save_plot=False):
        # _, gamma, thd = name.split('tensor')
        c1, c2 = self.evaluate_constraint(model)

        self.l_c.append((float(c1), float(c2)))

        if save_plot:
            if self.output_size > 1:
                for data, n in [data_t, data_v]:
                    f, ax = plt.subplots(self.output_size, 1, figsize=(30, 10) if n == 'train' else (10, 10))

                    data_x, data_y = DataCreater(data[0], data[1], self.input_size, self.output_size).creat_new_dataset(
                        seq_len=self.seq_len)
                    data_set = GetLoader(data_x, data_y)

                    data_set = DataLoader(data_set, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
                    predictions = torch.empty(1, self.output_size)

                    with torch.no_grad():
                        for batch_case, label in data_set:
                            label.to(self.device)
                            batch_case = batch_case.transpose(0, 1)
                            batch_case = batch_case.transpose(0, 2).to(torch.float32).to(self.device)
                            predict = model(batch_case).to(torch.float32).to(self.device)
                            predictions = torch.concat([predictions, predict.cpu()], dim=0)

                    i = 0
                    while i < self.output_size:
                        fit_score = self.nrmse(data_y[:, i], torch.tensor(predictions[1:, i]))
                        f.suptitle('Model: ' + path[18:-4] + 'c1:{} c2:{}'.format(c1, c2))
                        ax[i].plot(predictions[1:, i], color='m', label='pred', alpha=0.8)
                        ax[i].plot(data_y[:, i], color='c', label='real', linestyle='--', alpha=0.5)
                        ax[i].tick_params(labelsize=5)
                        ax[i].legend(loc='best')
                        ax[i].set_title('NRMSE on {} set: {:.3f}'.format(n, fit_score), fontsize=8)
                        i += 1
                    plt.savefig('./results{}.jpg'.format(path[6:-4] + n), bbox_inches='tight', dpi=500)
            else:
                f, ax = plt.subplots(2, 1)
                i = 0
                for data, n in [data_t, data_v]:
                    data_x, data_y = DataCreater(data[0], data[1], self.input_size,
                                                 self.output_size).creat_new_dataset(
                        seq_len=self.seq_len)
                    data_set = GetLoader(data_x, data_y)

                    data_set = DataLoader(data_set, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
                    predictions = torch.empty(1, self.output_size)

                    with torch.no_grad():
                        for batch_case, label in data_set:
                            label.to(self.device)
                            batch_case = batch_case.transpose(0, 1)
                            batch_case = batch_case.transpose(0, 2).to(torch.float32).to(self.device)
                            predict = model(batch_case).to(torch.float32).to(self.device)
                            predictions = torch.concat([predictions, predict.cpu()], dim=0)

                    fit_score = self.nrmse(data_y[:, :], torch.tensor(predictions[1:, :]))
                    f.suptitle('Model: ' + path[18:-4] + 'c1:{} c2:{}'.format(c1, c2))
                    ax[i].plot(predictions[1:, :], color='m', label='pred', alpha=0.8)
                    ax[i].plot(data_y[:, :], color='c', label='real', linestyle='--', alpha=0.5)
                    ax[i].tick_params(labelsize=5)
                    ax[i].legend(loc='best')
                    ax[i].set_title('NRMSE on {} set: {:.3f}'.format(n, fit_score), fontsize=8)
                    i += 1

                plt.savefig('./results{}.jpg'.format(path[6:-4] + n), bbox_inches='tight', dpi=500)

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
    # models = ['model_sl_5_bs_64_hs_5_ep_500_tol_1e-05_r_tensor([2, 2])_thd_tensor([1, 1]).pth']
    for model in models:
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
