#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : Features.py
@Author : chenbei
@Date : 2020/12/23 10:27
@Addess : https://www.cnblogs.com/cwp-bg/p/9488107.html
'''
'''
1、能量的表达式  : 对时间序列的每个点的值平方,然后求和 
2、对EMD熵来说首先对实现序列进行EMD分解,1个样本得到多个imf分量,例如得到1000×10(序列长度*imf分量个数)
   然后对每个imf分量(每列)求能量,即可以得到一个能量行向量(1×10) E = [E1,E2,E3,...,E10]
3、根据能量熵的定义需要先求得每个imf分量在总的imf分量的占比 Pe  1×10  即 Pe = [Pe1,Pe2,Pe3,...,Pe10]
4、能量熵公式为 He = -[Pe1*lg(Pe1)+Pe2*lg(Pe2)+,,,Pe10*lg(Pe10)] 最终1个样本得到1个能量熵
5、合并 241×1 的能量熵矩阵
6、类似的可以得到奇异熵，都是尺度分解 ,1000×10 处理
7、[近似熵、样本熵、排列熵、自相关系数、方差、均值] 不尺度分解 直接对样本的时间序列 列向量1000×1处理
8、上述合并为 1×8 特征矩阵
9、还需要定义一个函数,对所有样本进行处理 得到 241×8 特征矩阵
'''
import numpy as np
import pandas as pd
import pywt.data
from Feature_extraction.Approximate_entropy import ApEn
import matplotlib.pyplot as plt
class emd_entropy() :
    def __init__(self,array):
        self.array = array
    def get_energy(self):
        '''
        :param array: 时间序列长度*imf个数 参考1000*10
        :return: 行向量 1个样本的每个imf分量的能量 energy=[E1,E2,...Ek]
        '''
        energies = []
        for column in range(self.array.shape[1]):
            energies.append(sum(self.array[:, column] ** 2))
        return np.array(energies)

    def get_pe(self):
        # 返回每个能量的占比
        energy = self.get_energy()
        energysum = sum(energy)  # 求得不同imf的总能量 ,行向量
        new = energy / energysum  # 每个imf分量的占比 Pe = [Pe1,Pe2,...,Pek]
        return new
    def get_entropy(self):
        pe = self.get_pe() # [Pe1,Pe2,...,Pek]
        Entropy= 0
        for i in range(len(pe)) :
            Entropy = Entropy - pe[i] * np.log10(pe[i])
        return Entropy
class svd_entropy():
    def __init__(self,array):
        self.array = array
    def get_singular_value(self):
        '''
        一个 mxn的矩阵H可以分解为 U(mxm) ,S(mxn) , V(nxn) 三个矩阵的乘积，这就是奇异值分解
        S是一个对角矩阵，一般从大到小排列，S的元素值称为奇异值
        :return: 输入1000×10的单一样本数组 , 返回 奇异值 1×10 [svd1,svd2,...svd10]
        '''
        _, S, _ = np.linalg.svd(self.array) # U , S ,V = np.linalg.svd(example)
        return S
    def get_se(self):
        svd = self.get_singular_value()
        sumsvd = sum(svd) # 奇异值之和
        se = svd / sumsvd  # [Se1,Se2,...,Se10]
        return se
    def get_entropy(self):
        se  = self.get_se()
        Entropy = 0
        for i in range(len(se)):
            if se[i] == 0 :
                tem = 0
                print("存在奇异值为0,程序将跳过该元素防止出错")
            else:
                tem = se[i] * np.log10(se[i])
            Entropy = Entropy - tem
        return Entropy
class sample_entropy():
    '''
    # 与近似熵相比，样本熵具有两个优势：样本熵的计算不依赖数据长度；样本熵具有更好的一致性
    # 即参数m mm和r rr的变化对样本熵的影响程度是相同的.
    '''
    def __init__(self,vetor,m=2,r=0.2):
        self.vetor = vetor
        self.m = m
        self.r = r
        self.N = len(vetor)
    def get_maxdist(self,x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    def get_phi(self,m):
        x = [[self.vetor[j] for j in range(i, i + m - 1 + 1)] for i in range(self.N - m + 1)]
        B = [(len([1 for x_j in x if self.get_maxdist(x_i, x_j) <= self.r]) - 1.0) / (self.N - m) for x_i in x]
        return (self.N - m + 1.0) ** (-1) * sum(B)
    def get_entropy(self):
        return -np.log(self.get_phi(self.m+1) / self.get_phi(self.m))
class permutation_entropy():
    '''
    序列长度、嵌入维数和延迟时间
    m : 5~7比较合适 2~3太小, 12~15太大
    '''
    def __init__(self,vetor):
        self.vetor = vetor
    def get_func(self,n):
        """求阶乘"""
        if n == 0 or n == 1:
            return 1
        else:
            return (n * self.get_func(n - 1))
    def get_probality(self,S):
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
    def get_Entropy(self,m,t):
        """计算排列熵值"""
        length = len(self.vetor) - (m - 1) * t
        # 重构 k*m 矩阵
        y = [self.vetor[i:i + m * t:t] for i in range(length)]
        # 将各个分量升序排序
        S = [np.argsort(y[i]) for i in range(length)]
        # 计算每一种 m 维符号序列的概率
        freq_list = self.get_probality(S)
        # 计算排列熵
        pe = 0
        for freq in freq_list:
            pe += (- freq * np.log(freq))
        return pe / np.log(self.get_func(m))
    def plot_var_m(self):
        PE_m = []
        ms = [2, 3, 4, 5, 6, 7, 8]
        for m1 in ms:
            pe = self.get_Entropy(m=m1,t=3)  # 固定时延,改变嵌入维数
            PE_m.append(pe)
        plt.plot(PE_m)
        plt.show()
    def plot_var_t(self):
        PE_t = []
        ts = [1, 2, 3, 4, 5, 6, 7, 8]
        for t in ts:
            pe = self.get_Entropy(m=2,t=t)  # 固定时延,改变嵌入维数
            PE_t.append(pe)
        plt.plot(PE_t)
        plt.show()
class autocorrelation_coefficient():
    def __init__(self,vetor,k):
        self.vetor = vetor
        self.k = k
    def get_auto_corr(self):
        '''
        输入：时间序列timeSeries，滞后阶数k
        输出：时间序列timeSeries的k阶自相关系数
        '''
        l = len(self.vetor)
        # 取出要计算的两个数组
        timeSeries1 = self.vetor[0:l - self.k]
        timeSeries2 = self.vetor[self.k:]
        timeSeries_mean = self.vetor.mean()
        timeSeries_var = np.array([i ** 2 for i in self.vetor - timeSeries_mean]).sum()
        auto_corr = 0
        for i in range(l - self.k):
            temp = (timeSeries1[i] - timeSeries_mean) * (timeSeries2[i] - timeSeries_mean) / timeSeries_var
            auto_corr = auto_corr + temp
        return auto_corr
    def get_avarage(self):
        avarage = np.mean(self.vetor)
        return avarage
    def get_std(self):
        std = np.std(self.vetor)
        return std
data = pd.read_csv('C:/Users/chenbei/Desktop/钢/test2.csv')
X = data.iloc[:,1]
t = data.iloc[:,0]
#%%
if __name__ == '__main__':
    # %%
    # EMD 能量熵 尺度分解
    example = np.random.rand(1000, 10)  # 模拟的是单一样本的尺度分解特征矩阵
    EE = emd_entropy(example)
    ee = EE.get_energy()  # 单一样本的所有imf分量的能量
    Pe = EE.get_pe()  # 单一样本所有imf分量的比值
    EnergyEntropy = EE.get_entropy()
    # %%
    # 小波 奇异熵 尺度分解
    WE = svd_entropy(example)
    sv = WE.get_singular_value()
    Se = WE.get_se()
    SingularEntropy = WE.get_entropy()
    # %%
    # 近似熵 不尺度分解
    ap = ApEn(2, 0.2*np.var(X.values))  # 参数m:子集的大小一般取2 参数r:阀值基数,0.1~0.2 时间序列的方差
    ApproximateEntropy = ap.jinshishang(X.values) #
    # %%
    # 样本熵
    SE = sample_entropy(X.values, 2, 0.2*np.var(X))
    se = SE.get_entropy()
    # %%
    # 排列熵
    PE = permutation_entropy(X.values)
    pe = PE.get_Entropy(2, 3)
    PE.plot_var_m()
    PE.plot_var_t()
    # %%
    # 自相关系数
    data = pywt.data.ecg()
    AC = autocorrelation_coefficient(data, 1) #行向量
    Autocorrelation_coefficient = AC.get_auto_corr()
    std = AC.get_std()  # 方差
    ava = AC.get_avarage()  # 均值

