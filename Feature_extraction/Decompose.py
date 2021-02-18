#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : Decompose.py
@Author : chenbei
@Date : 2021/1/5 11:09
@Addess : https://www.cnblogs.com/cwp-bg/p/9488107.html
'''
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from matplotlib.pylab import mpl
plt.rcParams['font.sans-serif'] = ['Times New Roman'] # 设置字体风格,必须在前然后设置显示中文
mpl.rcParams['font.size'] = 10.5 # 图片字体大小
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文的命令
mpl.rcParams['axes.unicode_minus'] = False  #  显示负号的命令
plt.rcParams['figure.figsize'] = (7.8,6) # 设置figure_size尺寸
plt.rcParams['savefig.dpi'] = 600 # 图片像素
plt.rcParams['figure.dpi'] = 600 # 分辨率
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=10.5)
from pyhht import emd
import pandas as pd
import datetime
from pyentrp import entropy as ent #包 计算样本熵和排列熵
from vectorizedsampleentropy import vectsampen as vse # 包计算样本熵和近似熵
from vectorizedsampleentropy import vectapen
from Feature_extraction.Features import  emd_entropy , svd_entropy ,sample_entropy,permutation_entropy
# 能量熵、奇异熵、样本熵、排列熵,互相关熵、模糊熵在本文件中
from Feature_extraction.Approximate_entropy import ApEn ,NewApEn # Pincus近似熵、洪波算法的近似熵
def Sample_Entropy(x, m=2,r=0.2):
    """
    样本熵
    m 滑动时窗的长度
    r 阈值系数 取值范围一般为：0.1~0.25
    """
    # 将x转化为数组
    x = np.array(x)
    r = r*np.std(x)
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
def Fuzzy_Entropy(x, m=2, r=0.2, n=2):
    """
    模糊熵
    m 滑动时窗的长度
    r 阈值系数 取值范围一般为：0.1~0.25
    n 计算模糊隶属度时的维度
    """
    # 将x转化为数组
    x = np.array(x)
    r = r * np.std(x)
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
def plot_imfs(signal, imfs,title, time_samples=None, fignum=None):
    if time_samples is None:
        time_samples = np.arange(signal.shape[0])
    n_imfs = imfs.shape[0]
    plt.figure(num=fignum)
    axis_extent = max(np.max(np.abs(imfs[:-1, :]), axis=0))
    ax = plt.subplot(n_imfs + 1, 1, 1)
    ax.plot(time_samples, signal)
    ax.axis([time_samples[0], time_samples[-1], signal.min(), signal.max()])
    ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                   labelbottom=False)
    ax.grid(False)
    ax.set_ylabel('signal')
    ax.set_title(title)
    # Plot the IMFs
    for i in range(n_imfs - 1):
        print(i + 2)
        ax = plt.subplot(n_imfs + 1, 1, i + 2)
        ax.plot(time_samples, imfs[i, :])
        ax.axis([time_samples[0], time_samples[-1], -axis_extent, axis_extent])
        ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                       labelbottom=False)
        ax.grid(False)
        ax.set_ylabel('imf' + str(i + 1))
    # Plot the residue
    ax = plt.subplot(n_imfs + 1, 1, n_imfs + 1)
    ax.plot(time_samples, imfs[-1, :], 'r')
    ax.axis('tight')
    ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                   labelbottom=False)
    ax.grid(False)
    ax.set_ylabel('res')
    plt.show()
def Time_Features(x):
    x_peak = np.max(abs(x))  # 1、峰值
    x_mean = np.mean(x)  # 2、均值
    x_std = np.std(x)  # 3、标准差
    x_skew = np.mean((x - x_mean) ** 3) / pow(x_std, 3)  # 4、偏度
    x_kurt = np.mean((x - x_mean) ** 4) / pow(np.var(x), 2)  # 5、峭度
    x_rms = np.sqrt(pow(x_mean, 2) + pow(x_std, 2))  # 6、均方根
    x_fzz = x_peak / x_rms  # 7、峰值指标
    Features = np.array((x_skew ,x_kurt,x_fzz))
    return Features
def ApEnn(U, m=2, r=0.2):
    r = r * np.std(U)
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])
    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0)**(-1) * sum(np.log(C))
    N = len(U)
    return abs(_phi(m+1) - _phi(m))
def Approximate_Entropy(x, m=2, r=0.2):
    """
    近似熵
    m 滑动时窗的长度
    r 阈值系数 取值范围一般为：0.1~0.25
    """
    # 将x转化为数组
    x = np.array(x)
    r = r * np.std(x)
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
#%%
data = pd.read_csv('C:/Users/chenbei/Desktop/钢/test2.csv')
X = data.iloc[:,1]
t = data.iloc[:,0]
#%%
EMD = emd.EMD(X, t)
IMF = EMD.decompose()
#%% EMD可视化及频谱分析
plot_imfs(X.values, IMF,title='振动信号的EMD分解',time_samples=t.values)
#%%
legends = ['imf1分量','imf2分量','imf3分量','imf4分量']
for i in range(4) :
    N = 10000 # 采样频率
    imf = IMF[i,:]
    N1 = len(imf)
    fft_imf = fft(imf)
    f = np.arange(N1)
    abs_imf = np.abs(fft_imf )
    half_f = np.arange(0, N, N / N1)
    half_f = half_f[range(int(N1 / 2))]
    half_abs_imf = abs_imf[range(int(N1 / 2))]
    plt.plot(half_f, half_abs_imf  / (N1 / 2),label=legends[i])
    plt.legend()
    plt.title('本征模态分量的频谱图')
plt.show()
#%% 互相关系数熵
Corrlation = IMFCorr(X,IMF) # 得到每个imf分量与Corr的相关系数
P = Corrlation /  sum(Corrlation)
CorrEntropy = IMFCorrEntropy(P)
#%% 能量熵
EE = emd_entropy(IMF.T) # 实例化对象 , 只处理列形式的imf,先转置
ee = EE.get_energy()  # 单一样本的所有imf分量的能量
Pe = EE.get_pe()  # 单一样本所有imf分量的比值
EnergyEntropy = EE.get_entropy()
#%% 奇异熵
WE = svd_entropy(IMF.T) # 实例化求奇异熵的类 IMF是数组格式,14列60000行
sv = WE.get_singular_value() # 调用方法求奇异值
Se = WE.get_se() # 调用方法求归一化后的值
SingularEntropy = WE.get_entropy() # 求奇异熵
#%% 近似熵
Ap = NewApEn(2,0.2*np.std(X.values))
HongB0 = []
hongboap = Ap.hongbo_jinshishang(X.values) # 洪波近似熵 = 0.9859728
AAP = ApEn(2, 0.2*np.std(X.values))
ApproximateEntropy = AAP.jinshishang(X.values) # Pincus近似熵= 0.9859728
aap = ApEnn(X.values) # 函数1的近似熵= 0.705818
AAp = Approximate_Entropy(X.values) # 函数2的近似熵= 0.12871286
aapp = vectapen.apen(X.values,2,0.2*np.std(X.values)) # 调用包求近似熵= 0.70762183
'''
a = datetime.datetime.now()
for i in range(len(IMF)) : # 从结果可以看出后边的imf分量近似熵都是0，所以实际上求分量是没有意义的
    imf = IMF[i,:]
    hongbo = Ap.hongbo_jinshishang(imf.T)
    HongB0.append(hongbo)
b = datetime.datetime.now()
print(b-a) # 半分钟 1000数据点
'''
#%% 样本熵
SE = sample_entropy(X.values, 2, 0.2 * np.var(X))
se = SE.get_entropy()# 调用类求样本熵=1.0262358
samE = Sample_Entropy(X.values) # 调用函数求样本熵 = 0.125705
std_ts = np.std(X.values)
sample_entropy = ent.sample_entropy(X.values, 5, 0.2 * std_ts) # 调用包求样本熵2.23981、1.74636、0.83168、0.17080、0.02228
vsese = vse.sampen(X.values, m=2, r=0.2*std_ts) # 调用包求样本熵=0.8329198
#%% 排列熵
PE = permutation_entropy(X.values)
pe = PE.get_Entropy(5, 3) # 调用类求排列熵=0.839228
pppeee = ent.permutation_entropy(X.values,5,3) # 调用包求排列熵=5.796457
#%% 模糊熵
Fuzzy_Entropy =Fuzzy_Entropy(X)
#%% 可视化同一样本的不同imf能量熵、奇异值、互相关熵(样本熵、近似熵、排列熵不计算imf分量)
imfn = np.linspace(1,len(IMF),len(IMF))
plt.plot(imfn,Se,label='能量熵')
plt.plot(imfn,Pe,label='奇异熵')
plt.plot(imfn,P,label='互相关熵')
plt.xlabel('IMF分量')
plt.ylabel('信息熵')
plt.tight_layout()
plt.legend()
plt.show()
#%% 计算时域特征偏度、峭度、峰态系数
TimeFeatures = Time_Features(X)
#%%

