#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : 第五章的图.py
@Author : chenbei
@Date : 2020/12/25 13:16
'''
from matplotlib.pylab import mpl
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman'] # 设置字体风格,必须在前然后设置显示中文
mpl.rcParams['font.size'] = 10.5 # 图片字体大小
mpl.rcParams['font.sans-serif'] = ['SimHei','SongTi'] # 显示中文的命令
mpl.rcParams['axes.unicode_minus'] = False  #  显示负号的命令
plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
plt.rcParams['savefig.dpi'] = 600 # 图片像素
plt.rcParams['figure.dpi'] = 600 # 分辨率
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=10.5) # matplotlib内无中文字节码，需要自行手动添加
import pandas as pd
import numpy as np
def ChangeStyle(ax,ylabel,title) :
    ax.set_xlabel('x')
    ax.set_ylabel(ylabel)
    ax.legend((title,),loc='upper left')
    ax.set_title(title)
fig , ax = plt.subplots(2,2)
axs = ax.flatten()
x = np.arange(-5,5,0.001)
linear = x
tanh = np.tanh(x)
sigmoid = 1 / (1+np.exp(-x))
relu = [[] for _ in range(len(x))]
for i in range(len(x)) :
    if x[i] <=0 :
        relu[i] = 0
    else:
        relu[i] = x[i]
axs[0].plot(x,tanh)
axs[1].plot(x,sigmoid)
axs[2].plot(x,linear)
axs[3].plot(x,relu)
ylabels = ['f(x)=tanh(x)','f(x)=sigmoid(x)','f(x)=x','f(x)=relu(x)']
titles = ['双曲正切函数','Sigmoid函数','线性函数','Relu函数']
for i in range(len(axs)) :
    ChangeStyle(axs[i],ylabel=ylabels[i],title=titles[i])
plt.tight_layout()
plt.show()
