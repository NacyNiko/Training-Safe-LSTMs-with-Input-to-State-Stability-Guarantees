# -*- coding: utf-8 -*- 
# @Time : 2022/12/12 16:20 
# @Author : Yinan 
# @File : LSTM_train.py
import torch
from torch import nn
import pandas as pd
import torch.utils.data as Data
import matplotlib.pyplot as plt

""" define ISS LSTM """
class IssLSTM(nn.Module):
    def __init__(self,  input_size=1, hidden_size=5, output_size=1, num_layers=1):
        super(IssLSTM, self).__init__()
        self.hidden_size = hidden_size
        # define LSTM
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # define output layer
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, u, h, c):
        output_, (h, c) = self.lstm(u, (h, c))
        output_ = self.linear(output_)
        return output_, h, c

    """
    math::
        \begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}
    """

    def constraint(self):
        W_o = self.lstm.weight_ih_l0[-self.hidden_size:, :]
        U_o = self.lstm.weight_hh_l0[-self.hidden_size:, :]
        b_o = self.lstm.bias_ih_l0[-self.hidden_size:] + self.lstm.bias_hh_l0[-self.hidden_size:].unsqueeze(1)

        W_f = self.lstm.weight_ih_l0[self.hidden_size:2*self.hidden_size, :]
        U_f = self.lstm.weight_hh_l0[self.hidden_size:2*self.hidden_size, :]
        b_f = self.lstm.bias_ih_l0[self.hidden_size:2*self.hidden_size] + self.lstm.bias_hh_l0[self.hidden_size:2*self.hidden_size].unsqueeze(1)

        W_i = self.lstm.weight_ih_l0[:self.hidden_size, :]
        U_i = self.lstm.weight_hh_l0[:self.hidden_size, :]
        b_i = self.lstm.bias_ih_l0[:self.hidden_size] + self.lstm.bias_hh_l0[:self.hidden_size].unsqueeze(1)

        U_c = self.lstm.weight_hh_l0[2 * self.hidden_size: 3 * self.hidden_size, :]

        con1 = (1 + torch.sigmoid(torch.norm(torch.hstack((W_o, U_o, b_o)), torch.inf))) * \
               torch.sigmoid(torch.norm(torch.hstack((W_f, U_f, b_f)), torch.inf)) - 1
        con2 = (1 + torch.sigmoid(torch.norm(torch.hstack((W_o, U_o, b_o)), torch.inf))) * \
               torch.sigmoid(torch.norm(torch.hstack((W_i, U_i, b_i)), torch.inf)) * torch.norm(U_c, 1) - 1
        return torch.tensor([con1, con2])


def main(val=True):
    model = IssLSTM()
    """
    weight_ih_l[k] : the learnable input-hidden weights of the :math:`\text{k}^{th}` layer
                `(W_ii|W_if|W_ig|W_io)`, of shape `(4*hidden_size, input_size)` for `k = 0`.
                Otherwise, the shape is `(4*hidden_size, num_directions * hidden_size)`. If
                ``proj_size > 0`` was specified, the shape will be
                `(4*hidden_size, num_directions * proj_size)` for `k > 0`
            weight_hh_l[k] : the learnable hidden-hidden weights of the :math:`\text{k}^{th}` layer
                `(W_hi|W_hf|W_hg|W_ho)`, of shape `(4*hidden_size, hidden_size)`. If ``proj_size > 0``
                was specified, the shape will be `(4*hidden_size, proj_size)`.
            bias_ih_l[k] : the learnable input-hidden bias of the :math:`\text{k}^{th}` layer
                `(b_ii|b_if|b_ig|b_io)`, of shape `(4*hidden_size)`
            bias_hh_l[k] : the learnable hidden-hidden bias of the :math:`\text{k}^{th}` layer
                `(b_hi|b_hf|b_hg|b_ho)`, of shape `(4*hidden_size)`
    """
    model.train()

    """ optimizer: SGD"""
    optimizer = torch.optim.SGD(model.parameters(), lr=0.02)

    """ loss function: MSE """
    loss_ = nn.MSELoss()

    """ load training set """
    train_data = Data.TensorDataset(torch.FloatTensor(pd.read_csv(r'../ISS_LSTM_by_ipm/train/train_input_noise.csv', header=None).iloc[:, 1]),
                        torch.FloatTensor(pd.read_csv(r'../ISS_LSTM_by_ipm/train/train_output_noise.csv', header=None).iloc[:, 1]))
    train_loader = Data.DataLoader(dataset=train_data, batch_size=1, shuffle=False, drop_last=True)

    max_epoch = 5   # set num. of epochs
    loss_list = []  # save loss

    for epoch in range(max_epoch):
        print('epoch:{}'.format(epoch))
        epoch_loss = 0
        hidden_prev = torch.randn(1, 1, 5)
        cell_prev = torch.randn(1, 1, 5)

        if epoch == max_epoch - 1:
            outs = []
            for item in train_loader:
                batch_cases, batch_labels = item
                batch_cases = batch_cases.unsqueeze(1).unsqueeze(1)

                out, hidden_prev, cell_prev = model(batch_cases, hidden_prev, cell_prev)
                hidden_prev = hidden_prev.detach()
                cell_prev = cell_prev.detach()

                # hidden_state, cell_state = out[1]
                outs.append(float(out.reshape(-1, )))

                # r: weight  threshold
                r = torch.tensor([0., 0.])
                # threshold
                threshold = torch.tensor(0)

                loss = loss_(out, batch_labels) + torch.dot(r, torch.relu(model.constraint() + threshold))

                epoch_loss += loss.data.item()
                loss_list.append(loss.data.item())

                """ Back propagation """
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        else:
            for item in train_loader:
                batch_cases, batch_labels = item
                batch_cases = batch_cases.unsqueeze(1).unsqueeze(1)

                out, hidden_prev, cell_prev = model(batch_cases, hidden_prev, cell_prev)
                hidden_prev = hidden_prev.detach()
                cell_prev = cell_prev.detach()

                # r: weight  threshold
                r = torch.tensor([0., 0.])
                # threshold
                threshold = torch.tensor(0)

                loss = loss_(out, batch_labels) + torch.dot(r, torch.relu(model.constraint() + threshold))

                epoch_loss += loss.data.item()
                loss_list.append(loss.data.item())

                """ Back propagation """
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        print('epoch:{},loss:{}'.format(epoch, epoch_loss))

    """ save model """
    torch.save(model.state_dict(), 'model.pth')


    fig, ax = plt.subplots(2, 1)
    """ plot last train epoch """
    ax[0].plot(range(4400), outs, label='prediction')
    ax[0].plot(range(4400), pd.read_csv(r'../ISS_LSTM_by_ipm/train/train_output_noise.csv', header=None).iloc[:, 1], label = 'ground truth')
    ax[0].set_title('last epoch')
    ax[0].legend()

    if val:
        model.eval()
        """ validation """
        u = pd.read_csv(r'../ISS_LSTM_by_ipm/train/train_input_noise.csv', header=None).iloc[:, 1]
        y = pd.read_csv(r'../ISS_LSTM_by_ipm/train/train_output_noise.csv', header=None).iloc[:, 1]

        """ load training set """
        train_data = Data.TensorDataset(
            torch.FloatTensor(u),
            torch.FloatTensor(y)
        )

        train_loader = Data.DataLoader(dataset=train_data, batch_size=1, shuffle=False, drop_last=False)
        y_pred = []

        """ initialize hidden state & cell state """
        hidden_prev = torch.randn(1, 5)
        cell_prev = torch.randn(1, 5)

        with torch.no_grad():
            for x, label in train_loader:
                y_hat, hidden_prev, cell_prev = model(x.unsqueeze(0), hidden_prev, cell_prev)
                y_pred.append(y_hat.detach().numpy().reshape(1,))

        ax[1].plot(range(4400), y_pred, label='prediction')
        ax[1].plot(range(4400), y, label='ground truth')
        ax[1].set_title('train set on trained model')
        ax[1].legend()

    plt.show()


if __name__ == '__main__':
    main()

"""
Q: why does the trained model even on train set seem pretty different from last epoch?
"""