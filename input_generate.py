# -*- coding: utf-8 -*- 
# @Time : 2022/11/15 14:05 
# @Author : Yinan 
# @File : input_generate.py
import random

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def input_generate(T, num_pulse):

    """ [low, high] num. of pulse """
    low = num_pulse[0]
    high = num_pulse[1]

    np.random.seed(100)
    num_pulse = np.random.randint(low, high)
    a = T//high - T//80
    b = T//low + T//80

    """ set length for each pulse """
    len_pulse = np.random.randint(a, b, num_pulse, dtype='int64')

    """ initialize input u and disturbance d"""
    initial_input = 15.6
    u = np.zeros(T)
    d = np.zeros(T)
    length = 0
    multiple = 2

    while length < T:
        for i in len_pulse:
            u[length:min(T, length+i)] = (np.random.rand(1)-0.5) * multiple + initial_input   # delta_u = (np.random.rand(1)-0.5) * multiple
            for j in range(length, min(T, length+i)):
                d[j] = 0.005*np.random.randn()
            length += i
    return u + d


# u, d = input_generate(, [25, 35])
# pd.DataFrame(u).to_excel('input_data.xlsx')
# pd.DataFrame(d).to_excel('noise_data.xlsx')
# plt.plot(np.arange(2000), u+d)
# plt.show()
