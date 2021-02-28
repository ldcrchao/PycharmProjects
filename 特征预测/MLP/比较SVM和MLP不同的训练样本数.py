#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : 比较SVM和MLP不同的训练样本数.py
@Author : chenbei
@Date : 2020/12/25 18:36
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
import pandas as pd
import numpy as np
from time import time
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
def mlp_acu_onetraining(clf,X,y,train_size=0.7) :
    # 一次训练得到的准确率 可以用来作为基础函数被反复调用
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=train_size)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    num_x_train = len(y_train) # 返回训练样本数量
    true_num = 0 # 预测正确的数量
    for i in range(len(y_pred)) :
        if y_pred[i] == y_test[i] :
            true_num = true_num + 1
    acu = true_num / len(y_pred) # 单次预测的准确率
    return acu ,num_x_train
def get_mlp_acu_trainsize(clf,maxiter,train_size,X,y) :
    # 获得某个transize下的准确率ACU 反复调用基础函数
    ACU = []
    num_X_train = []
    for i in range(maxiter): # 训练代数改变
        acu , num_x_train = mlp_acu_onetraining(clf, X, y, train_size=train_size) # 这是训练一次得到的准确率
        ACU.append(acu)
        num_X_train.append(num_x_train)
    return ACU ,num_X_train
def plot_mlp_acu_trainsize(ACU_X,ACU,time,maxiter) :
    fig ,ax = plt.subplots()
    ax.plot(ACU_X,ACU, 'g-v', linewidth=2,markersize=2, label='平均准确率') #折线图
    ax.plot([1, max(ACU_X)],[min(ACU), min(ACU)],'r-o', label='平均准确率最小值', linewidth=1) # 直线
    ax.plot([1, max(ACU_X)],[np.mean(ACU), np.mean(ACU)], 'b-o' ,label='平均准确率平均值', linewidth=1) # 直线

    ax.fill_between(np.arange(1, max(ACU_X), 1),
                    np.mean(ACU) + np.std(ACU), np.mean(ACU) - np.std(ACU),
                    alpha=0.1,color='r')
    ax.text((1 + max(ACU_X)) / 2, np.mean(ACU) + 0.005,
            "准确率平均值 : " + str(round(np.mean(ACU), 5)),
             horizontalalignment='center',color='b', fontsize=16)
    ax.text((1 + max(ACU_X)) / 2, min(ACU) + 0.005,
            "准确率最小值 : " + str(round(min(ACU), 5)),
            horizontalalignment='center',color='r', fontsize=16)
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
def ChangeMLPTrainsize(X,y):
    # 调用固定transize的maxiter次准确率---->调用基础函数求1次准确率
    trainsizes = np.arange(0.1, 1.0, 0.1)
    clf = MLPClassifier(activation= 'identity',solver='lbfgs', alpha=0.1,hidden_layer_sizes=(5, 2), random_state=1)
    maxiter = 100
    ACUmean = []
    ACU_X = []
    Time = []
    for trainsize in trainsizes:  # 不同训练样本数
        starttime = time()
        acu, acu_x = get_mlp_acu_trainsize(clf, maxiter, trainsize, X=X, y=y)  # 某一个训练比例,迭代maxiter次 得到相应的acu
        acu_mean = np.mean(acu)  # 找到每个训练比例下迭代maiter次的准确率平均值
        ACUmean.append(acu_mean)
        acu_x_mean = np.mean(acu_x)
        ACU_X.append(acu_x_mean)  # 每次训练的数量 这里平均都是一样的
        endtime = time()
        consumetime = endtime - starttime
        Time.append(consumetime)
    Time = np.array(Time) / maxiter  # 归算到每一次花费的时间
    plot_mlp_acu_trainsize(ACU_X, ACUmean, Time, maxiter=maxiter)
def plot_svm_acu_trainsize(ACU_X,ACU,time,maxiter) :
    fig ,ax = plt.subplots()
    ax.plot(ACU_X,ACU, 'g-v', linewidth=2,markersize=2, label='平均准确率')
    ax.plot([1, max(ACU_X)],[min(ACU), min(ACU)],'r-o', label='平均准确率最小值', linewidth=1)
    ax.plot([1, max(ACU_X)],[np.mean(ACU), np.mean(ACU)], 'b-o' ,label='平均准确率平均值', linewidth=1)

    ax.fill_between(np.arange(1, max(ACU_X), 1), np.mean(ACU) + np.std(ACU), np.mean(ACU) - np.std(ACU), alpha=0.1,
                     color='r')
    ax.text((1 + max(ACU_X)) / 2, np.mean(ACU) + 0.005, "准确率平均值 : " + str(round(np.mean(ACU), 5)),
             horizontalalignment='center',color='b', fontsize=16)
    ax.text((1 + max(ACU_X)) / 2, min(ACU) + 0.005, "准确率最小值 : " + str(round(min(ACU), 5)), horizontalalignment='center',color='r', fontsize=16)
    ax.legend(loc='upper right')
    ax.set_ylabel('平均准确率')
    ax.set_xlabel('训练样本数量/个')
    ax1 = ax.twinx()
    ax1.plot(ACU_X ,time,'c-d',linewidth=2,label='训练时间')
    ax1.set_ylabel('每次训练时间/s')
    ax1.legend()
    plt.title(f'支持向量机SVM每{maxiter}次的平均准确率')
    plt.legend(loc='upper left')
    plt.show()
def get_svm_acu(clf,X,y,maxiter,trainsize) :
    # 每一次划分训练集后应该迭代100次
    ACU = []
    for i in range(maxiter) :
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=trainsize)
        clf.fit(X_train, y_train)  # 训练模型
        y_pre = clf.predict(X_test) # 某一次的预测结果
        true_num = 0
        for j in range(len(y_pre)) :
            if y_pre[j] == y_test[j] :
                true_num = true_num + 1
        acu = true_num / len(y_pre) # 某一次的预测准确率
        ACU.append(acu) # 100次的所有预测准确率
    return ACU
def get_svm_acu_trainsize(clf,X,y,maxiter,trainsizes) :
    Mean_ACU_Trainsize = [] # 不同训练比例下的100次平均准确率 11个值
    Mean_Time_Trainsize = []
    Train_num = []
    for trainsize in trainsizes :
        train_num = len(X) * trainsize
        A = time()
        ACU_Trainsize = get_svm_acu(clf,X,y,maxiter,trainsize) # 某一个训练比例下100次的准确率
        B = time()
        time_testsize = round(B-A,3) / maxiter
        Mean_ACU = np.mean(ACU_Trainsize) # 取100次的平均
        Mean_ACU_Trainsize.append(Mean_ACU)
        Mean_Time_Trainsize.append(time_testsize)
        Train_num.append(train_num)
    return Mean_ACU_Trainsize ,Mean_Time_Trainsize,Train_num # 每个比例100次准确率的平均值、平均时间、该比例对应训练集个数
def ChangeSVMTrainsize(X,y) :
    testsizes = np.arange(0.1, 1.0, 0.1)
    clf=svm.SVC(kernel='linear',C=1,probability=True)
    maxiter = 100
    acu,time,acu_x = get_svm_acu_trainsize(clf,X,y,maxiter,testsizes) # 导入模型、迭代次数、训练比例
    plot_svm_acu_trainsize(acu_x,acu,time,maxiter )
Data = pd.read_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/FirstLevelPCA.csv",encoding='gbk')
X_dataframe = Data.iloc[:,0:-1] # 分出数据和标签 此时是DataFrame格式
y_dataframe = Data.iloc[:,-1]
X = X_dataframe.values # ndarray格式 样本数×维数
y_category = y_dataframe.values # ndarray格式
Label = LabelEncoder() # 初始化1个独热编码类
y = Label.fit_transform(y_category) # 自动生成标签
#%%
# 改变训练集比例来对比MLP和SVM模型的准确率和花费时间
ChangeMLPTrainsize(X,y)
ChangeSVMTrainsize(X,y)
