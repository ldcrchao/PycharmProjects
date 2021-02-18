#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : MLP.py
@Author : chenbei
@Date : 2020/12/22 19:35
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
import numpy as np
from time import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
def MLP_NeuralNetwork(clf,X,y,train_size=0.3) :
    '''
    :param clf: 模型
    :param X: 训练样本
    :param y: 训练标签
    :param test_size: 固定的比例
    :return: 训练一次时的准确率和此比例下的训练样本数,用于后续画图,固定比例可以忽略返回的样本数
    '''
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=train_size)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    numX_train = len(y_train) # 返回训练样本数量
    #y_proba = clf.predict_proba(X_test)
    sum = 0
    for i in range(len(y_pred)) :
        if y_pred[i] == y_test[i] :
            sum = sum + 1
    #print('准确率为 : ' , round(sum/len(y_pred)*100,5),'%')
    sum = sum /len(y_pred)
    return sum ,numX_train
def get_ACU(clf,maxiter,train_size,X,y) :
    '''
    :param clf: 模型
    :param maxiter: 训练次数
    :param test_size: 固定的比例
    :return: 每次训练的准确率
    '''
    ACU = []
    num_X_train = []
    for i in range(maxiter): # 训练代数改变
        acu , num_x_train = MLP_NeuralNetwork(clf, X, y, train_size=train_size) # 这是训练一次得到的准确率
        ACU.append(acu)
        num_X_train.append(num_x_train)
    return ACU ,num_X_train
def PlotACU(ACU,maxiter,trainsize) :
    plt.plot(ACU, 'c-p', linewidth=1,markersize=2, label='准确率')
    plt.plot([1, maxiter], [min(ACU), min(ACU)], 'r-o',label='准确率最小值', linewidth=1)
    plt.plot([1, maxiter], [np.mean(ACU), np.mean(ACU)], 'b-o',label='准确率平均值', linewidth=1 )
    plt.fill_between(np.arange(1, maxiter, 1), np.mean(ACU) + np.std(ACU), np.mean(ACU) - np.std(ACU), alpha=0.1,
                     color='r')
    plt.text((1 + maxiter) / 2, np.mean(ACU) + 0.005, "准确率平均值 : " + str(round(np.mean(ACU), 5))
             ,horizontalalignment='center',color='b', fontsize=16)
    plt.text((1 + maxiter) / 2, min(ACU) + 0.005, "准确率最小值 : " + str(round(min(ACU), 5)), horizontalalignment='center',color='r', fontsize=16)
    #plt.text((1 + maxiter) / 2, (min(ACU) + max(ACU)) / 2-0.04, "Std ACU : " + str(round(np.std(ACU), 5)),
    #         horizontalalignment='center', family="Times New Roman", fontsize=16)
    #plt.text((1 + maxiter) / 2, (min(ACU) + max(ACU)) / 2 - 0.04, "Training Time : " + str(round(time, 5)),
    #         family="Times New Roman", fontsize=16)
    plt.title(f'MLP神经网络准确率变化图(训练集比例:{trainsize})')
    plt.ylabel('准确率')
    plt.xlabel('训练次数')
    plt.legend(loc='lower left')
    plt.show()
def PlotACUMulTestSize(ACU_X,ACU,time,maxiter) :
    '''
    :param ACU_X: 样本数量 list
    :param ACU: 不同样本数量对应的maxiter次平均准确率
    :param time: 不同样本数量对应的maxiter次 时间
    :return: 预测准确率和消耗时间 趋势图
    '''
    fig ,ax = plt.subplots()
    ax.plot(ACU_X,ACU, 'g-v', linewidth=2,markersize=2, label='平均准确率')
    ax.plot([1, max(ACU_X)],[min(ACU), min(ACU)],'r-o', label='平均准确率最小值', linewidth=1)
    ax.plot([1, max(ACU_X)],[np.mean(ACU), np.mean(ACU)], 'b-o' ,label='平均准确率平均值', linewidth=1)
    ax.fill_between(np.arange(1, max(ACU_X), 1), np.mean(ACU) + np.std(ACU), np.mean(ACU) - np.std(ACU), alpha=0.1,
                     color='r')
    ax.text((1 + max(ACU_X)) / 2, np.mean(ACU) + 0.005, "准确率平均值 : " + str(round(np.mean(ACU), 5)),
             horizontalalignment='center',color='b', fontsize=16)
    ax.text((1 + max(ACU_X)) / 2, min(ACU) + 0.005, "准确率最小值 : " + str(round(min(ACU), 5)), horizontalalignment='center',color='r', fontsize=16)
    #ax.text((1 + max(ACU_X)) / 2, (min(ACU) + max(ACU)) / 2, "Std ACU : " + str(round(np.std(ACU), 5)),
    #         family="Times New Roman", fontsize=16)
    ax.legend(loc='upper right')
    ax.set_ylabel('平均准确率')
    ax.set_xlabel('训练样本数量/个')
    ax1 = ax.twinx()
    ax1.plot(ACU_X ,time,'c-d',linewidth=2,label='训练时间')
    ax1.set_ylabel('每次训练时间/s')
    ax1.legend()
    plt.title(f'MLP神经网络每{maxiter}次的平均准确率')
    plt.legend(loc='upper left')
    plt.show()
Data = pd.read_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/FirstLevelPCA.csv",encoding='gbk')
X_dataframe = Data.iloc[:,0:-1] # 分出数据和标签 此时是DataFrame格式
y_dataframe = Data.iloc[:,-1]
X = X_dataframe.values # ndarray格式 样本数×维数
y_category = y_dataframe.values # ndarray格式
Label = LabelEncoder() # 初始化1个独热编码类
y = Label.fit_transform(y_category) # 自动生成标签
#%%
# 固定样本数量 不同训练次数 0 ~100次
clf = MLPClassifier(activation= 'identity',solver='lbfgs', alpha=0.1,hidden_layer_sizes=(5, 2), random_state=1) #表示2层,第一层5个神经元 第二层2个
maxiter = 100
trainsize = 0.7
ACU,_ = get_ACU(clf,maxiter,trainsize,X=X,y=y)
PlotACU(ACU,maxiter =maxiter,trainsize=trainsize)
#%%
# 考虑不同训练样本数量的准确率,固定训练次数100次
clf1 = MLPClassifier(activation= 'identity',solver='lbfgs', alpha=0.1,hidden_layer_sizes=(5, 2), random_state=1)
testsizes = np.arange(0.1,1.0,0.1)
maxiter = 100
ACUmean = []
ACU_X = []
Time = []
for testsize in testsizes : # 不同训练样本数
    starttime = time()
    acu , acu_x = get_ACU(clf1,maxiter,testsize,X,y)  # 某一个训练比例,迭代maxiter次 得到相应的acu
    acu_mean = np.mean(acu) # 找到每个训练比例下迭代maiter次的准确率平均值
    ACUmean.append(acu_mean)
    acu_x_mean = np.mean(acu_x)
    ACU_X.append(acu_x_mean) # 每次训练的数量
    endtime = time()
    consumetime = endtime - starttime
    Time.append(consumetime)
Time = np.array(Time) / maxiter # 归算到每一次花费的时间
PlotACUMulTestSize(ACU_X,ACUmean,Time,maxiter=maxiter)
#%%


