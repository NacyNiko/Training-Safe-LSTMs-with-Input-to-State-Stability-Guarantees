# -*- coding: utf-8 -*- 
# @Time : 2022/11/15 14:05 
# @Author : Yinan 
# @File : input_generate.py
import numpy as np


def input_generate(T, num_pulse):
    """ [low, high] num. of pulse """
    low = num_pulse[0]
    high = num_pulse[1]

    num_pulse = np.random.randint(low, high)
    a = T//high - T//80
    b = T//low + T//80

    """ set length for each pulse """
    len_pulse = np.random.randint(a, b, num_pulse, dtype='int64')

    """ initialize input u and disturbance d"""
    initial_input = 15.6
    u = np.zeros(T)

    length = 0
    multiple = 2

    while length < T:
        for i in len_pulse:
            u[length:min(T, length+i)] = (np.random.rand(1)-0.5) * multiple + initial_input   # delta_u = (np.random.rand(1)-0.5) * multiple
            # for j in range(length, min(T, length+i)):
            #     d[j] = 0.003*np.random.randn()
            length += i
    return u
