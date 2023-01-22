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
from utilities import DataCreater, GetLoader, normalize
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
    def __init__(self,  r, threshold, device='cpu'):
        self.input_size = None
        self.hidden_size = None
        self.output_size = None
        self.num_layers = None
        self.device = device

        self.l_r = np.array(sum(list([i] * len(r) for i in range(0, len(r))), []))
        self.l_thd = np.array(list(range(0, len(threshold))) * len(threshold))
        self.l_c = []
    @staticmethod
    def load_data():
        data_t, n_t = [r'../data/train/train_input_noise.csv', r'../data/train/train_output_noise.csv'], 'train'
        data_v, n_v = [r'../data/val/val_input_noise.csv', r'../data/val/val_output_noise.csv'], 'val'
        return [data_t, n_t], [data_v, n_v]

    @staticmethod
    def nrmse(y, y_hat):  # normalization to y
        y = y.squeeze(1)
        return np.sqrt(1 / y.shape[0]) * torch.norm(y - y_hat)

    def create_model(self, input_size, hidden_size, output_size, num_layers):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        model = LstmRNN(input_size, hidden_size, output_size, num_layers)
        return model

    def load_model(self, model, path):
        model.load_state_dict(torch.load(path))
        model.eval()
        model.to(self.device)
        return model

    def evaluate(self, model, name, data_t, data_v, seq_len=5, save_plot=False):
        _, r, thd = name.split('tensor')
        c1, c2 = self.evaluate_constraint(model)

        self.l_c.append((float(c1), float(c2)))

        if save_plot:
            f, ax = plt.subplots(2, 1)
            i = 0
            for data, n in [data_t, data_v]:
                data_x, data_y = DataCreater(data[0], data[1]).creat_new_dataset(
                    seq_len=seq_len)
                data_set = GetLoader(data_x, data_y)

                data_set = DataLoader(data_set, batch_size=1, shuffle=False, drop_last=False, num_workers=0)
                predictions = list()

                with torch.no_grad():
                    for batch_case, label in data_set:
                        label.to(self.device)
                        batch_case = batch_case.transpose(0, 1)
                        batch_case = batch_case.transpose(0, 2).to(torch.float32).to(self.device)
                        predict = model(batch_case).to(torch.float32).to(self.device)
                        predictions.append(predict.squeeze(0).squeeze(0).cpu())

                fit_score = self.nrmse(data_y, torch.tensor(predictions))
                f.suptitle('Model: ' + name[18:-4] + 'c1:{} c2:{}'.format(c1, c2))
                ax[i].plot(predictions, color='m', label='pred', alpha=0.8)
                ax[i].plot(data_y, color='c', label='real', linestyle='--', alpha=0.5)
                ax[i].tick_params(labelsize=5)
                ax[i].legend(loc='best')
                ax[i].set_title('NRMSE on {} set: {:.3f}'.format(n, fit_score), fontsize=8)
                i += 1
            plt.savefig('./results{}.jpg'.format(name[8:-4]), bbox_inches='tight', dpi=500)


    def evaluate_constraint(self, model):
        def constraint(paras, hidden_size=self.hidden_size, r=None, threshold=None):  # paras = model.lstm.parameters()  [W, U, b1, b2]
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
            return con1, con2
        parameters = model.lstm.parameters()
        c1, c2 = constraint(parameters)
        return c1, c2


def main(if_filter=True):   # if_filter: ignore whether r=0 or threshold=0
    validator = Validator([2], [1], device='cuda')
    data_train, data_val = validator.load_data()
    lstmmodel = validator.create_model(2, 5, 1, 1)
    file = './models/'
    # models = os.listdir(file)
    models = ['model_sl_5_bs_64_hs_5_ep_100_tol_1e-05_r_tensor([2, 2])_thd_tensor([1, 1])____.pth']
    for model in models:
        path = file + model
        lstmmodel = validator.load_model(lstmmodel, path)
        validator.evaluate(lstmmodel, path, data_train, data_val, save_plot=True)

    if if_filter:
        idx = validator.l_r * validator.l_thd
        r = validator.l_r[idx != 0]
        thd = validator.l_thd[idx != 0]
        c = pd.DataFrame(validator.l_c)[idx != 0]
    else:
        r = validator.l_r
        thd = validator.l_thd
        c = pd.DataFrame(validator.l_c)

    ax_ = plt.axes(projection='3d')
    ax_.scatter3D(r, thd, c.iloc[:, 0], c=c.iloc[:, 0], s=500*normalize(c.iloc[:, 1]) if if_filter else 100)  # min: -0.9983   max:-0.9955
                                                                                        # smaller dot: more negative

    ax_.set_xlabel('ratio')
    ax_.set_ylabel('threshold')
    ax_.set_zlabel('c1')
    plt.show()
    print('-------------Finish---------------------')


if __name__ == '__main__':
    main()
