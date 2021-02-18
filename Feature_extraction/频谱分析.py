#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : 频谱分析.py
@Author : chenbei
@Date : 2021/1/5 11:16
'''
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import pandas as pd
import numpy as np
from matplotlib.pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文的命令
mpl.rcParams['axes.unicode_minus'] = False  #  显示负号的命令
N = 20000 # 采样率
f1 = 5000 # 信号最大频率
x=np.linspace(0,1,N)
y = np.cos(2*np.pi*f1*x)
fft_y=fft(y)
f = np.arange(N)
abs_y=np.abs(fft_y)
half_f = f[range(int(N/2))]
half_abs_y = abs_y[range(int(N/2))]
plt.plot(half_f,half_abs_y / (N/2))
plt.show()

x1=x[0:1000]   # 前1000个点 时域缩小20倍 ，那频域步长扩大20倍
N1 = len(x1)
y = np.cos(2*np.pi*f1*x1)
fft_y=fft(y)
f = np.arange(N1)
abs_y=np.abs(fft_y)
half_f = np.arange(0,N,N/N1)
half_f = half_f [range(int(N1/2))]
half_abs_y = abs_y[range(int(N1/2))]
plt.plot(half_f,half_abs_y / (N1/2))
plt.show()