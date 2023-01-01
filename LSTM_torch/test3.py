# -*- coding: utf-8 -*- 
# @Time : 2022/12/22 1:01 
# @Author : Yinan 
# @File : test3.py
import scipy.io as sio
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
from myPackage import DataCreater, GetLoader
from torch.utils.data import DataLoader

"""
输入： 2 维度，[t-w:t]的y 以及 [t]的u
输出： lstm最后一层隐状态经过Linear层的值

训练结果记录：
    1. 神经元取10， 序列长度5， 最终误差1e-4时过拟合
    2. 神经元取5， 序列长度5， 最终误差1e-3时欠拟合
    2. 神经元取5， 序列长度10， 最终误差1e-3时欠拟合
    
"""
# Define LSTM Neural Networks
class LstmRNN(nn.Module):

    def __init__(self, input_size, hidden_size=5, output_size=1, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  #  [seq_len, batch_size, input_size] --> [seq_len, batch_size, hidden_size]
        self.linear1 = nn.Linear(hidden_size, output_size)  #  [seq_len, batch_size, hidden_size] --> [seq_len, batch_size, output_size]

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = x.view(s, b, -1)
        return x[-1, :, :]


if __name__ == '__main__':
    """ hyperparameter """
    train = False
    seq_len = 10
    batch_size = 64
    hidden_size = 5
    INPUT_FEATURES_NUM = 2
    OUTPUT_FEATURES_NUM = 1

    # checking if GPU is available
    device = torch.device("cpu")
    use_GPU = torch.cuda.is_available()

    if use_GPU:
        longTensor = torch.cuda.LongTensor
        floatTensor = torch.cuda.FloatTensor
        device = torch.device("cuda:0")
        print('Training on GPU.')
    else:
        longTensor = torch.LongTensor
        floatTensor = torch.FloatTensor
        print('No GPU available, training on CPU.')



    if train:
        train_x, train_y = DataCreater('train_x.csv', 'train_y.csv').creat_new_dataset(seq_len=seq_len)
        train_set = GetLoader(train_x, train_y)
        train_set = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)

        # ----------------- train -------------------
        lstm_model = LstmRNN(INPUT_FEATURES_NUM, hidden_size, output_size=OUTPUT_FEATURES_NUM, num_layers=1)
        lstm_model.to(device)
        print('LSTM model:', lstm_model)
        print('model.parameters:', lstm_model.parameters)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)

        prev_loss = 1000
        max_epochs = 3000
        break_flag = False
        for epoch in range(max_epochs):
            for batch_cases, labels in train_set:
                batch_cases = batch_cases.transpose(0, 1)
                batch_cases = batch_cases.transpose(0, 2).to(torch.float32).to(device)
                output = lstm_model(batch_cases).to(torch.float32).to(device)
                labels = labels.to(torch.float32).to(device)

                loss = criterion(output, labels)

                """ backpropagation """
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if loss < prev_loss:
                    torch.save(lstm_model.state_dict(), 'lstm_model.pt')  # save model parameters to files
                    prev_loss = loss

                if loss.item() < 1e-4:
                    break_flag = True
                    print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
                    print("The loss value is reached")
                    break
                elif (epoch + 1) % 100 == 0:
                    print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))
            if break_flag:
                break
        torch.save(lstm_model.state_dict(), 'test3_04.pth')

    else:
        """ eval """
        net = LstmRNN(2, hidden_size, output_size=1, num_layers=1)
        net.load_state_dict(torch.load('test3_04.pth'))
        net.eval()
        net.to(device)

        test_x, test_y = DataCreater('test_x.csv', 'test_y.csv').creat_new_dataset(seq_len=seq_len)
        test_set = GetLoader(test_x, test_y)

        test_set = DataLoader(test_set, batch_size=1, shuffle=False, drop_last=False, num_workers=2)
        predictions = list()

        with torch.no_grad():
            for batch_case, label in test_set:
                label.to(device)
                batch_case = batch_case.transpose(0, 1)
                batch_case = batch_case.transpose(0, 2).to(torch.float32).to(device)
                predict = net(batch_case).to(torch.float32).to(device)
                predictions.append(predict.squeeze(0).squeeze(0).cpu().numpy())


        plt.plot(predictions, 'r', label='pred')
        plt.plot(test_y, 'b', label='real', alpha=0.3)
        plt.legend(loc='best')
        plt.show()


