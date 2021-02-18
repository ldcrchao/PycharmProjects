#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : 奈斯特罗和蒙特卡洛近似.py
@Author : chenbei
@Date : 2020/12/27 13:44
'''
from matplotlib.pylab import mpl
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman'] # 设置字体风格,必须在前然后设置显示中文
mpl.rcParams['font.size'] = 10.5 # 图片字体大小
mpl.rcParams['font.sans-serif'] = ['SimHei','SongTi'] # 显示中文的命令
mpl.rcParams['axes.unicode_minus'] = False  #  显示负号的命令
plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
plt.rcParams['savefig.dpi'] = 600 # 图片像素
plt.rcParams['figure.dpi'] = 400 # 分辨率
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=10.5) # matplotlib内无中文字节码，需要自行手动添加
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.kernel_approximation import (RBFSampler,Nystroem) # 内核近似
import pandas as pd
import numpy as np
from sklearn import pipeline
from time import time
from sklearn.model_selection import train_test_split # 留出法的分割方式
Data = pd.read_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/FirstLevelPCA.csv",encoding='gbk')
X_dataframe = Data.iloc[:,0:-1] # 分出数据和标签 此时是DataFrame格式
y_dataframe = Data.iloc[:,-1]
X = X_dataframe.values # ndarray格式 样本数×维数
y_category = y_dataframe.values # ndarray格式
Label = LabelEncoder() # 初始化1个独热编码类
y = Label.fit_transform(y_category) # 自动生成标签
'''蒙特卡洛近似和奈斯特罗姆近似'''
X = pd.read_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/Circuit_Breaker_FirstLevelLabel.csv",encoding='gbk')
X = X.drop(['Category'],axis=1) # 删除标签列
X = X.values # 划分训练和测试的参数是数组格式
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1)
n_spilit = len(X_train)
n_spilit_size = np.linspace(2,n_spilit,n_spilit-1)  # 2,3,4,...192 至少2份
sample_sizes =  np.arange(1,n_spilit)
kernel_svm = svm.SVC(gamma=.1) # 不指定默认是rbf
linear_svm = svm.LinearSVC() # 与指定'linear'是一样的
Linear_time = []
Kernel_time = []
Linear_score = []
Kernel_score = []
starttime = time()
for i in range(len(n_spilit_size)) : # 依次使用训练集的2,3,4,...192个样本
    # 以下四行程序是为了保证每次使用训练集其中的i个样本也是随机的
    state = np.random.get_state() # 必须先打乱原来的数据集合标签集,否则初始的有序状态会导致交叉验证预测率很低
    np.random.shuffle(X_train)
    np.random.set_state(state) # 保证样本和标签以相同的规律被打乱
    np.random.shuffle(y_train)
    #print(int(n_spilit_size[i]))
    X_train_son = X_train[0:int(n_spilit_size[i]),:] # 前 i 行 作为训练集 的子集
    y_train_son = y_train[0:int(n_spilit_size[i])] # 前 i 行 作为训练集 的子集
    Kernel_svm_time = time() # rbf核函数的程序开始时间
    kernel_svm.fit(X_train_son, y_train_son) # 训练集拟合
    kernel_svm_score = kernel_svm.score(X_test, y_test) # 测试集得分
    kernel_svm_time = time() - Kernel_svm_time # rbf核函数运行消耗的时间
    Linear_svm_time = time() # 线性核函数的程序开始时间
    linear_svm.fit(X_train, y_train) # 训练集拟合
    linear_svm_score = linear_svm.score(X_test, y_test) # 测试集得分
    linear_svm_time = time() - Linear_svm_time # 线性核函数运行消耗的时间
    Linear_score.append(linear_svm_score)  # 存放不同训练子集数量预测测试集的时间和分数
    Linear_time.append(linear_svm_time)
    Kernel_score.append(kernel_svm_score)
    Kernel_time.append(kernel_svm_time)
'''线性核 : 从内核近似创建管道'''
feature_map_fourier = RBFSampler(gamma=.2, random_state=1) # 傅里叶变换的蒙特卡洛近似
feature_map_nystroem = Nystroem(gamma=.2, random_state=1) # 使用数据的子集为任意内核构造一个近似特征图
fourier_approx_svm_rbf = pipeline.Pipeline([("feature_map", feature_map_fourier), ("svm", svm.SVC(gamma=.1))]) # 基于RBF构建第一个管道
nystroem_approx_svm_rbf = pipeline.Pipeline([("feature_map", feature_map_nystroem),("svm", svm.SVC(gamma=.1))]) # 基于RBF构建第二个管道
fourier_approx_svm_linear = pipeline.Pipeline([("feature_map", feature_map_fourier), ("svm", svm.LinearSVC())]) # 基于Linear构建第一个管道
nystroem_approx_svm_linear = pipeline.Pipeline([("feature_map", feature_map_nystroem),("svm", svm.LinearSVC())]) # 基于Linear构建第二个管道
fourier_scores_rbf = [] # 分别存放fourier和nystroem内核估计的分数和消耗时间
nystroem_scores_rbf = []
fourier_times_rbf = []
nystroem_times_rbf = []
fourier_scores_linear = [] # 分别存放fourier和nystroem内核估计的分数和消耗时间
nystroem_scores_linear = []
fourier_times_linear = []
nystroem_times_linear = []
for D in sample_sizes:
    # 基于RBF
    fourier_approx_svm_rbf.set_params(feature_map__n_components=D) # 管道1设置参数 ,D不能超过训练集的数量
    nystroem_approx_svm_rbf.set_params(feature_map__n_components=D)
    start = time()
    nystroem_approx_svm_rbf.fit(X_train, y_train) # rbf的nystroem时间
    nystroem_times_rbf.append(time() - start)
    start = time()
    fourier_approx_svm_rbf.fit(X_train, y_train)# rbf的fourier时间
    fourier_times_rbf.append(time() - start)
    fourier_score = fourier_approx_svm_rbf.score(X_test, y_test) # rbf的nystroem得分
    nystroem_score = nystroem_approx_svm_rbf.score(X_test, y_test) # rbf的fourier得分
    nystroem_scores_rbf.append(nystroem_score)
    fourier_scores_rbf.append(fourier_score)
    # 基于Linear
    fourier_approx_svm_linear.set_params(feature_map__n_components=D) # 管道1设置参数 ,D不能超过训练集的数量
    nystroem_approx_svm_linear.set_params(feature_map__n_components=D)
    start = time()
    nystroem_approx_svm_linear.fit(X_train, y_train) # rbf的nystroem时间
    nystroem_times_linear.append(time() - start)
    start = time()
    fourier_approx_svm_linear.fit(X_train, y_train)# rbf的fourier时间
    fourier_times_linear.append(time() - start)
    fourier_score = fourier_approx_svm_linear.score(X_test, y_test) # rbf的nystroem得分
    nystroem_score = nystroem_approx_svm_linear.score(X_test, y_test) # rbf的fourier得分
    nystroem_scores_linear.append(nystroem_score)
    fourier_scores_linear.append(fourier_score)
plt.figure(figsize=(12, 8.4))
accuracy = plt.subplot(211) # 准确率
timescale = plt.subplot(212) # 耗时
# 基于不同核函数的两种估计的时间和准确率
accuracy.plot(sample_sizes, nystroem_scores_rbf, label="基于RBF核的奈斯特罗姆近似--准确率")
timescale.plot(sample_sizes, nystroem_times_rbf, '--',label='基于RBF核的奈斯特罗姆近似-模型训练时间')
accuracy.plot(sample_sizes, fourier_scores_rbf, label="基于RBF核的蒙特卡洛近似-准确率")
timescale.plot(sample_sizes, fourier_times_rbf, '--',label='基于RBF核的蒙特卡洛近似-模型训练时间')
accuracy.plot(sample_sizes, nystroem_scores_linear, label="基于Linear核的奈斯特罗姆近似-准确率")
timescale.plot(sample_sizes, nystroem_times_linear, '--',label='基于Linear核的奈斯特罗姆近似-模型训练时间')
accuracy.plot(sample_sizes, fourier_scores_linear, label="基于Linear核的蒙特卡洛近似-准确率")
timescale.plot(sample_sizes, fourier_times_linear, '--',label='基于Linear核的蒙特卡洛近似-模型训练时间')
# 没有近似的线性核和RBF核的时间和准确率
accuracy.plot(sample_sizes,Linear_score, label="普通线性核-准确率")
timescale.plot(sample_sizes,Linear_time,'--', label='普通线性核-模型训练时间')
accuracy.plot(sample_sizes,Kernel_score, label="普通RBF核-准确率")
timescale.plot(sample_sizes,Kernel_time,'--', label='普通RBF核-模型训练时间')
# 设置图片参数
minscore  = min(np.min(fourier_scores_rbf),np.min(nystroem_scores_rbf),
                np.min(Kernel_score),np.min(Linear_score),
                np.min(fourier_scores_linear),np.min(nystroem_scores_linear)) #找到6种评价分数各自最小的最小那个作为y轴最小值
maxscore = max(np.max(fourier_scores_rbf),np.max(nystroem_scores_rbf),
                np.max(Kernel_score),np.max(Linear_score),
                np.max(fourier_scores_linear),np.max(nystroem_scores_linear))
accuracy.set_ylim(minscore, maxscore)
accuracy.set_xlim(0,len(sample_sizes))
accuracy.set_ylabel("准确率")
accuracy.set_xlabel("训练集样本数量/个")
accuracy.legend(loc='best')
accuracy.set_title("基于Linear和RBF核的奈斯特罗姆近似与蒙特卡洛近似的准确率变化图")
mintime =  min(np.min(fourier_times_rbf),np.min(nystroem_times_rbf),
               np.min(Kernel_time),np.min(Linear_time),
               np.min(fourier_times_linear),np.min(nystroem_times_linear))
maxtime = max(np.max(fourier_times_rbf),np.max(nystroem_times_rbf),
              np.max(Kernel_time),np.max(Linear_time),
              np.max(fourier_times_linear),np.max(nystroem_times_linear))
timescale.set_ylim(mintime, maxtime)
timescale.set_xlim(0,len(sample_sizes))
timescale.set_xlabel("训练集样本数量/个")
timescale.set_ylabel("模型训练时间/s")
timescale.legend(loc='best')
timescale.set_title("基于Linear和RBF核的奈斯特罗姆近似与蒙特卡洛近似的模型训练时间变化图")
plt.show()
endtime = time()
print('程序花费时间为 : ',endtime - starttime)