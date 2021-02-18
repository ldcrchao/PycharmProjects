#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : 五个熵.py
@Author : chenbei
@Date : 2021/1/5 18:41
@Address : https://blog.csdn.net/husky66/article/details/112512532
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from matplotlib.pylab import mpl
plt.rcParams['font.sans-serif'] = ['Times New Roman'] # 设置字体风格,必须在前然后设置显示中文
mpl.rcParams['font.size'] = 10.5 # 图片字体大小
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文的命令
mpl.rcParams['axes.unicode_minus'] = False  #  显示负号的命令
plt.rcParams['figure.figsize'] = (7.8,3.8) # 设置figure_size尺寸
plt.rcParams['savefig.dpi'] = 600 # 图片像素
plt.rcParams['figure.dpi'] = 600 # 分辨率
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=10.5)
import pandas as pd
import datetime
import pywt
def Fuzzy_Entropy(x, m=2, r=0.25, n=2):
    """
    模糊熵
    m 滑动时窗的长度
    r 阈值系数 取值范围一般为：0.1~0.25
    n 计算模糊隶属度时的维度
    """
    # 将x转化为数组
    x = np.array(x)
    #r = r * np.var(x)
    # 检查x是否为一维数据
    if x.ndim != 1:
        raise ValueError("x的维度不是一维")
    # 计算x的行数是否小于m+1
    if len(x) < m+1:
        raise ValueError("len(x)小于m+1")
    # 将x以m为窗口进行划分
    entropy = 0  # 近似熵
    for temp in range(2):
        X = []
        for i in range(len(x)-m+1-temp):
            X.append(x[i:i+m+temp])
        X = np.array(X)
        # 计算X任意一行数据与其他行数据对应索引数据的差值绝对值的最大值
        D_value = []  # 存储差值
        for index1, i in enumerate(X):
            sub = []
            for index2, j in enumerate(X):
                if index1 != index2:
                    sub.append(max(np.abs(i-j)))
            D_value.append(sub)
        # 计算模糊隶属度
        D = np.exp(-np.power(D_value, n)/r)
        # 计算所有隶属度的平均值
        Lm = np.average(D.ravel())
        entropy = abs(entropy) - Lm

    return entropy

def Approximate_Entropy(x, m=2, r=0.2):
    """
    近似熵
    m 滑动时窗的长度
    r 阈值系数 取值范围一般为：0.1~0.25
    """
    # 将x转化为数组
    x = np.array(x)
    r = r * np.var(x)
    # 检查x是否为一维数据
    if x.ndim != 1:
        raise ValueError("x的维度不是一维")
    # 计算x的行数是否小于m+1
    if len(x) < m+1:
        raise ValueError("len(x)小于m+1")
    # 将x以m为窗口进行划分
    entropy = 0  # 近似熵
    for temp in range(2):
        X = []
        for i in range(len(x)-m+1-temp):
            X.append(x[i:i+m+temp])
        X = np.array(X)
        # 计算X任意一行数据与所有行数据对应索引数据的差值绝对值的最大值
        D_value = []  # 存储差值
        for i in X:
            sub = []
            for j in X:
                sub.append(max(np.abs(i-j)))
            D_value.append(sub)
        # 计算阈值
        F = r*np.std(x, ddof=1)
        # 判断D_value中的每一行中的值比阈值大的个数除以len(x)-m+1的比例
        num = np.sum(D_value>F, axis=1)/(len(x)-m+1-temp)
        # 计算num的对数平均值
        Lm = np.average(np.log(num))
        entropy = abs(entropy) - Lm
    return entropy
def Sample_Entropy(x, m=2,r=0.2):
    """
    样本熵
    m 滑动时窗的长度
    r 阈值系数 取值范围一般为：0.1~0.25
    """
    # 将x转化为数组
    x = np.array(x)
    r = r*np.var(x)
    # 检查x是否为一维数据
    if x.ndim != 1:
        raise ValueError("x的维度不是一维")
    # 计算x的行数是否小于m+1
    if len(x) < m+1:
        raise ValueError("len(x)小于m+1")
    # 将x以m为窗口进行划分
    entropy = 0  # 近似熵
    for temp in range(2):
        X = []
        for i in range(len(x)-m+1-temp):
            X.append(x[i:i+m+temp])
        X = np.array(X)
        # 计算X任意一行数据与所有行数据对应索引数据的差值绝对值的最大值
        D_value = []  # 存储差值
        for index1, i in enumerate(X):
            sub = []
            for index2, j in enumerate(X):
                if index1 != index2:
                    sub.append(max(np.abs(i-j)))
            D_value.append(sub)
        # 计算阈值
        F = r*np.std(x, ddof=1)
        # 判断D_value中的每一行中的值比阈值大的个数除以len(x)-m+1的比例
        num = np.sum(D_value>F, axis=1)/(len(X)-m+1-temp)
        # 计算num的对数平均值
        Lm = np.average(np.log(num))
        entropy = abs(entropy) - Lm
    return entropy
def Corr(a,b) :
    A = pd.Series(a)
    B = pd.Series(b)
    Corr =round(A.corr(B),5)
    return Corr
def IMFCorr(signal,IMF) :
    IMF = IMF.T
    Corraltion = []
    for  i in range(IMF.shape[1]) :
        imf = IMF[:,i]
        corr = Corr(imf,signal)
        Corraltion.append(abs(corr))
    return Corraltion
def IMFCorrEntropy(corr) :
    entopy = 0
    for i in range(len(corr)) :
        if corr[i] == 0 :
            print('存在互相关系数为0,程序将跳过该元素防止出错')
            tem = 0
        else:
            tem = corr[i] * np.log10(corr[i])
        entopy = entopy  + tem
    entopy = entopy  * (-1)
    return entopy
def func(n):
    """求阶乘"""
    if n == 0 or n == 1:
        return 1
    else:
        return (n * func(n - 1))
def compute_p(S):
    """计算每一种 m 维符号序列的概率"""
    _map = {}
    for item in S:
        a = str(item)
        if a in _map.keys():
            _map[a] = _map[a] + 1
        else:
            _map[a] = 1

    freq_list = []
    for freq in _map.values():
        freq_list.append(freq / len(S))
    return freq_list
def Permutation_Entropy(x, m=6, t=3):
    """计算排列熵值"""
    length = len(x) - (m - 1) * t
    # 重构 k*m 矩阵
    y = [x[i:i + m * t:t] for i in range(length)]
    # 将各个分量升序排序
    S = [np.argsort(y[i]) for i in range(length)]
    # 计算每一种 m 维符号序列的概率
    freq_list = compute_p(S)
    # 计算排列熵
    pe = 0
    for freq in freq_list:
        pe += (- freq * np.log(freq))
    return pe / np.log(func(m))
data = pd.read_csv('C:/Users/chenbei/Desktop/钢/test2.csv')
X = data.iloc[:,1]
t = data.iloc[:,0]
#%% 样本熵
a = datetime.datetime.now()
Sample_Entropy = Sample_Entropy(X)
print(Sample_Entropy)
b = datetime.datetime.now()
print(b-a)
#%% 近似熵
a = datetime.datetime.now()
Approximate_Entropy = Approximate_Entropy(X)
print(Approximate_Entropy)
b = datetime.datetime.now()
print(b-a)
#%% 模糊熵
a = datetime.datetime.now()
Fuzzy_Entropy =Fuzzy_Entropy(X)
print(Fuzzy_Entropy)
b = datetime.datetime.now()
print(b-a)
#%% 排列熵
Permutation_Entropy(X.values)
#%% https://blog.csdn.net/Cratial/article/details/79707169 近似熵
def ApEn(U, m=2, r=0.2):
    r = r * np.var(U)
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))

    N = len(U)

    return abs(_phi(m+1) - _phi(m))

data = pd.read_csv('C:/Users/chenbei/Desktop/钢/test2.csv')
X = data.iloc[:,1]
t = data.iloc[:,0]
ApEn(X.values)




