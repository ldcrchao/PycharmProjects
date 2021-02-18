#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : 留出法.py
@Author : chenbei
@Date : 2020/12/23 20:45
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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import svm
import pandas as pd
import numpy as np
from time import time
def onetrain(clf,X,y,testsize):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=testsize)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    numX_train = len(y_train)  # 返回训练样本数量
    sum = 0
    for i in range(len(y_pred)):
        if y_pred[i] == y_test[i]:
            sum = sum + 1
    sum = sum / len(y_pred) # 准确率
    return sum ,numX_train
def maxitertrain(clf,X,y,maxiter,testsize) :
    ACU = []
    num_X_train = [] # 固定样本数时此参数忽略
    for i in range(maxiter):  # 训练代数改变 如100次训练的准确率 需要ACU存储
        acu, num_x_train = onetrain(clf, X, y, testsize=testsize)  # 这是训练一次得到的准确率
        ACU.append(acu)
        num_X_train.append(num_x_train)
    return ACU, num_X_train
def PlotACU(ACU,testsize,title,maxiter) :
    plt.plot(ACU, 'c-p', linewidth=1,markersize=2, label='准确率')
    plt.plot([1, maxiter], [min(ACU), min(ACU)], 'r-o',label='准确率最小值', linewidth=1, )
    plt.plot([1, maxiter], [np.mean(ACU), np.mean(ACU)],'b-o', label='准确率平均值', linewidth=1)
    plt.fill_between(np.arange(1, maxiter, 1), np.mean(ACU) + np.std(ACU), np.mean(ACU) - np.std(ACU), alpha=0.1,
                     color='r')
    plt.text((1 + maxiter) / 2, np.mean(ACU) + 0.01, "Avarage ACU : " + str(round(np.mean(ACU), 5)),
             horizontalalignment='center', family="Times New Roman", fontsize=16)
    plt.text((1 + maxiter) / 2, min(ACU) + 0.01, "Min ACU : " + str(round(min(ACU), 5)), family="Times New Roman",
             horizontalalignment='center',fontsize=16)
    plt.text((1 + maxiter) / 2, (min(ACU) + max(ACU)) / 2 -0.04, "Std ACU : " + str(round(np.std(ACU), 5)),
             horizontalalignment='center', family="Times New Roman", fontsize=16)
    plt.title(title+f'准确率变化图(测试集比例:{testsize})')
    plt.ylabel('准确率')
    plt.xlabel('训练次数')
    plt.legend(loc='lower left')
    plt.show()
def PlotACUMulTestSize(ACU_X,ACU,time,title,maxiter) :
    '''
    :param ACU_X: 样本数量 list
    :param ACU: 不同样本数量对应的maxiter次平均准确率
    :param time: 不同样本数量对应的maxiter次 时间
    :return: 预测准确率和消耗时间 趋势图
    '''
    fig ,ax = plt.subplots()
    ax.plot(ACU_X,ACU, 'g-v', linewidth=2,markersize=2, label='平均准确率')
    ax.plot([1, max(ACU_X)],[min(ACU), min(ACU)],'r-o', label='平均正确率最小值', linewidth=1)
    ax.plot([1, max(ACU_X)],[np.mean(ACU), np.mean(ACU)], 'b-o' ,label='平均正确率平均值', linewidth=1)
    ax.fill_between(np.arange(1, max(ACU_X), 1), np.mean(ACU) + np.std(ACU), np.mean(ACU) - np.std(ACU), alpha=0.1,
                     color='r')
    ax.text((1 + max(ACU_X)) / 2, np.mean(ACU) + 0.02, "Avarage ACU : " + str(round(np.mean(ACU), 5)),
             horizontalalignment='center',family="Times New Roman", fontsize=16)
    ax.text((1 + max(ACU_X)) / 2, min(ACU) + 0.02, "Min ACU : " + str(round(min(ACU), 5)), family="Times New Roman",
            horizontalalignment='center', fontsize=16)
    ax.text((1 + max(ACU_X)) / 2, (min(ACU) + max(ACU)) / 2, "Std ACU : " + str(round(np.std(ACU), 5)),
             family="Times New Roman",horizontalalignment='center', fontsize=16)
    ax.legend(loc='upper right')
    ax.set_ylabel('平均准确率')
    ax.set_xlabel('训练样本数量/个')
    ax1 = ax.twinx()
    ax1.plot(ACU_X ,time,'c-d',linewidth=2,label='训练时间')
    ax1.set_ylabel('每次训练时间/s')
    ax1.legend()
    plt.title(title+f'每{maxiter}次的平均准确率')
    plt.legend(loc='upper left')
    plt.show()
def FixedSampleSize(kernel,title,X,y) :
    maxiter = 240
    testsize = 0.3
    clf=svm.SVC(kernel=kernel,C=1,probability=True)
    #oneacu = onetrain(clf,X,y,testsize) # 训练一次的准确率是不够的
    manyacus,_ = maxitertrain(clf,X=X,y=y,maxiter=maxiter,testsize=testsize) # 固定样本数时此参数忽略
    PlotACU(manyacus,testsize,title,maxiter=maxiter)
def FixedNumberOfTraining(kernel,title,X,y) :
    clf1 = svm.SVC(kernel=kernel,C=1,probability=True)
    testsizes = np.arange(0.1,1.0,0.1)
    maxiter = 240
    ACUmean = []
    ACU_X = []
    Time = []
    for testsize in testsizes : # 不同训练样本数
        starttime = time()
        acu , acu_x = maxitertrain(clf1,X,y,maxiter,testsize)  # 某一个训练比例,迭代maxiter次 得到相应的acu 和对应的样本数
        acu_mean = np.mean(acu) # 找到每个训练比例下迭代maiter次的准确率平均值 这是因为固定训练次数的话需要1个比例训练多次所以要取平均
        ACUmean.append(acu_mean)
        acu_x_mean = np.mean(acu_x)
        ACU_X.append(acu_x_mean) # 每次训练的数量
        endtime = time()
        consumetime = endtime - starttime
        Time.append(consumetime)
    Time = np.array(Time) / maxiter # 归算到每一次花费的时间
    PlotACUMulTestSize(ACU_X,ACUmean,Time,title,maxiter)
Data = pd.read_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/FirstLevelPCA.csv",encoding='gbk')
X_dataframe = Data.iloc[:,0:-1] # 分出数据和标签 此时是DataFrame格式
y_dataframe = Data.iloc[:,-1]
X = X_dataframe.values # ndarray格式 样本数×维数
y_category = y_dataframe.values # ndarray格式
Label = LabelEncoder() # 初始化1个独热编码类
y = Label.fit_transform(y_category) # 自动生成标签
#%% 固定样本数量 不同训练次数 0~100次
FixedSampleSize(kernel='linear',title='SVM线性核函数',X=X,y=y)
FixedSampleSize(kernel='rbf',title='SVM径向基核函数',X=X,y=y)
#%%
# 考虑不同训练样本数量的准确率,固定训练次数100次
FixedNumberOfTraining(kernel='rbf',title='SVM线性核函数',X=X,y=y)
FixedNumberOfTraining(kernel='rbf',title='SVM径向基核函数',X=X,y=y)