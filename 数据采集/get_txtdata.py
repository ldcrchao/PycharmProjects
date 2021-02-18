#%%
# -*- coding UTF-8 -*-
'''
@Project : MyProjects
@File : get_txtdata.py
@Author : chenbei
@Date : 2021/1/21 15:27
'''
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
plt.rcParams['font.sans-serif'] = ['Times New Roman'] # 设置字体风格,必须在前然后设置显示中文
mpl.rcParams['font.size'] = 10.5 # 图片字体大小
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文的命令
mpl.rcParams['axes.unicode_minus'] = False  #  显示负号的命令
mpl.rcParams['agg.path.chunksize'] = 10000
plt.rcParams['figure.figsize'] = (7.8,3.8) # 设置figure_size尺寸
plt.rcParams['savefig.dpi'] = 600 # 图片像素
plt.rcParams['figure.dpi'] = 600 # 分辨率
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=10.5)
import numpy as np
import pandas as pd
import datetime
time = []
alm = []
with open("C:\\Users\\chenbei\\Desktop\\钢\\data.txt",encoding='utf-8') as f :
     for i, line in enumerate(f.readlines()):
         reads = line.strip().split("||||")
         time.append(reads[0])
         alm.append(reads[1])
     f.close()
# 将time的str格式转换为datetime格式方便计算
Time = []
for i in range(len(time)):
    # 同时都减去开头的时间,将时间初始化为0
    tem = datetime.datetime.strptime(time[i], '%Y-%m-%d %H:%M:%S.%f') -datetime.datetime.strptime(time[0], '%Y-%m-%d %H:%M:%S.%f')
    # 取出初始化后时间的秒和毫秒
    tem = tem.seconds + tem.microseconds * 0.000001
    Time.append(tem)
Alm = pd.DataFrame((list(map(float,alm)) ))# 将字符型的元素变为浮点型
T = pd.DataFrame(Time)
Data = pd.concat([T,Alm],axis=1)
Data.columns=['时间','幅值']
#Data.to_csv("C:\\Users\\chenbei\\Desktop\\钢\\txt的csv.csv",index=False) #保存数
#plt.plot(T.values,Alm.values)# 可视化
#plt.show()
#%%
