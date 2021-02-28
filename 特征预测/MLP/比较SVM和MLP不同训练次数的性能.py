#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : 比较SVM和MLP不同训练次数的性能.py
@Author : chenbei
@Date : 2020/12/25 20:14
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
def mlp_acu_onetraining(clf,X,y,train_size=0.3) :
    # 一次训练得到的准确率 可以用来作为基础函数被反复调用
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=train_size)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    num_x_train = len(y_train) # 返回训练样本数量
    true_num = 0
    for i in range(len(y_pred)) :
        if y_pred[i] == y_test[i] :
            true_num = true_num + 1
    acu = true_num /len(y_pred)# 单次预测的准确率
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
def plot_mlp_acu(ACU,maxiter,trainsize) :
    plt.plot(ACU, 'c-p', linewidth=1,markersize=2, label='准确率')
    plt.plot([1, maxiter], [min(ACU), min(ACU)], 'r-o',label='准确率最小值', linewidth=1)
    plt.plot([1, maxiter], [np.mean(ACU), np.mean(ACU)], 'b-o',label='准确率平均值', linewidth=1 )
    plt.fill_between(np.arange(1, maxiter, 1),
                     np.mean(ACU) + np.std(ACU), np.mean(ACU) - np.std(ACU),
                     alpha=0.1,color='r')
    plt.text((1 + maxiter) / 2, np.mean(ACU) + 0.005,
             "准确率平均值 : " + str(round(np.mean(ACU), 5))
             ,horizontalalignment='center',color='b', fontsize=16)
    plt.text((1 + maxiter) / 2, min(ACU) + 0.005,
             "准确率最小值 : " + str(round(min(ACU), 5)),
             horizontalalignment='center',color='r', fontsize=16)
    plt.title(f'MLP神经网络准确率变化图(训练集比例:{trainsize})')
    plt.ylabel('准确率')
    plt.xlabel('训练次数')
    plt.legend(loc='lower left')
    plt.show()
def get_svm_acu(clf,X,y,maxiter,trainsize=0.7) :
    # 每一次划分训练集后应该迭代100次
    PP = []
    for i in range(maxiter) :
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=trainsize)
        clf.fit(X_train, y_train)  # 训练模型
        y_pre = clf.predict(X_test) # 某一次的预测结果
        pp = 0
        for j in range(len(y_pre)) :
            if y_pre[j] == y_test[j] :
                pp = pp + 1
        pp = pp / len(y_pre) # 某一次的预测准确率
        PP.append(pp) # 100次的所有预测准确率
    return PP ,maxiter ,trainsize
def plot_svm_acu(ACU,maxiter,trainsize) :
    plt.plot(ACU, 'c-p', linewidth=1,markersize=2, label='准确率')
    plt.plot([1, maxiter], [min(ACU), min(ACU)], 'r-o',label='准确率最小值', linewidth=1)
    plt.plot([1, maxiter], [np.mean(ACU), np.mean(ACU)], 'b-o',label='准确率平均值', linewidth=1 )
    plt.fill_between(np.arange(1, maxiter, 1), np.mean(ACU) + np.std(ACU), np.mean(ACU) - np.std(ACU), alpha=0.1,
                     color='r')
    plt.text((1 + maxiter) / 2, np.mean(ACU) + 0.005, "准确率平均值 : " + str(round(np.mean(ACU), 5))
             ,horizontalalignment='center',color='b', fontsize=16)
    plt.text((1 + maxiter) / 2, min(ACU) + 0.005, "准确率最小值 : " + str(round(min(ACU), 5)), horizontalalignment='center',color='r', fontsize=16)
    plt.title(f'支持向量机准确率变化图(训练集比例:{trainsize})')
    plt.ylabel('准确率')
    plt.xlabel('训练次数')
    plt.legend(loc='lower left')
    plt.show()
Data = pd.read_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/FirstLevelPCA.csv",encoding='gbk')
X_dataframe = Data.iloc[:,0:-1] # 分出数据和标签 此时是DataFrame格式
y_dataframe = Data.iloc[:,-1]
X = X_dataframe.values # ndarray格式 样本数×维数
y_category = y_dataframe.values # ndarray格式
Label = LabelEncoder() # 初始化1个独热编码类
y = Label.fit_transform(y_category) # 自动生成标签
#%%
# 随着训练次数的增加来比较2个最佳模型的准确率和花费时间
clf_mlp = MLPClassifier(activation= 'identity',solver='lbfgs',
                    alpha=0.1,hidden_layer_sizes=(5, 2), random_state=1) #表示2层,第一层5个神经元 第二层2个
maxiter = 100
trainsize = 0.7
# 获得某个训练集比例的准确率
ACU,_ = get_mlp_acu_trainsize(clf_mlp,maxiter,trainsize,X=X,y=y)
plot_mlp_acu(ACU,maxiter =maxiter,trainsize=trainsize)

clf_svm = svm.SVC(kernel='linear', C=1, probability=True)
acu , maxiter,trainsize = get_svm_acu(clf_svm,X,y,maxiter=100,trainsize=0.7)
plot_svm_acu(acu,maxiter,trainsize)
#%%
