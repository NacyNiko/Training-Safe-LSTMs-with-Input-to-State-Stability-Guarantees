# -*- coding: utf-8 -*- 
# @Time : 2022/12/21 0:59 
# @Author : Yinan 
# @File : data_generate.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
a, b = 125, 180
x = np.arange(125, 180)
y = x ** (1/2) + 0.5*np.sin(x) + np.random.randn(b-a) * 0.005

plt.figure()
plt.plot(range(len(y)), y)
plt.show()


pd.DataFrame(x).to_csv(r'./test_x.csv')
pd.DataFrame(y).to_csv(r'./test_y.csv')