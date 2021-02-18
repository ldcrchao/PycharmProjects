#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : 测试.py
@Author : chenbei
@Date : 2020/12/23 10:14
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Feature_extraction.Approximate_entropy import ApEn
from Feature_extraction.Features import emd_entropy ,svd_entropy , sample_entropy ,permutation_entropy,autocorrelation_coefficient
data = pd.read_csv('C:/Users/chenbei/Desktop/钢/线圈数据-20k/数据-1100点/220.csv')
'''对于近似、排列、样本、方差、均值只对样本处理,data模拟的是样本'''
#%% 获得1个样本的能量熵 (此时data被看成1个样本的多个imf分量) 如果循环,应当是循环读取文件夹的所有文件然后处理
eemd = emd_entropy(data.values)
EMD_Entropy  = eemd.get_entropy()
#%% 获得一个样本的奇异熵 与EMD熵同理
svd = svd_entropy(data.values)
SVD_Entropy = svd.get_entropy()
#%% 循环获得每个样本的样本熵 此时data被看成多个样本,每列是1个样本
Sample_Entropy = []
for i in [1,2]:
    columndata = data.iloc[:,i].values
    #columndata = columndata.reshape(-1,1)
    Sam = sample_entropy(columndata)
    ed = Sam.get_entropy()
    Sample_Entropy.append(ed)
#%% 循环获得每个样本的近似熵
Approxiate_Entropy = []
for i in [1,2]:
    columndata = data.iloc[:,i].values
    #columndata = columndata.reshape(-1,1)
    Sam = ApEn(2,0.2)
    ed = Sam.jinshishang(columndata)
    Approxiate_Entropy.append(ed)
#%% 循环获得每个样本的排列熵
Permutation_Entropy = []
for i in [1,2]:
    columndata = data.iloc[:,i].values
    #columndata = columndata.reshape(-1,1)
    PE = permutation_entropy(columndata)
    ed = PE.get_Entropy(4,2)
    Permutation_Entropy.append(ed)
#%%
Auto_Coeffs = []
AVA = []
STD = []
for i in [1,2]:
    columndata = data.iloc[:,i].values
    #columndata = columndata.reshape(-1,1)
    PE = autocorrelation_coefficient(columndata,2) # 自相关系数中需要行向量
    ed = PE.get_auto_corr()
    ava = PE.get_avarage()
    std = PE.get_std()
    Auto_Coeffs.append(ed)
    AVA.append(ava)
    STD.append(std)
