#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : SVM_CompareKernels-5_8.py
@Author : chenbei
@Date : 2020/12/17 16:11
'''
from matplotlib.pylab import mpl
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman'] # 设置字体风格,必须在前然后设置显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文的命令
mpl.rcParams['axes.unicode_minus'] = False  #  显示负号的命令
from sklearn import svm,datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1 # 得到最小值和最大值加减1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), # 得到二维网格图
                         np.arange(y_min, y_max, h))
    return xx, yy
def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out , Z
#iris = datasets.load_iris()
#X = iris.data # 样本数×维数 ndarray
#y = iris.target # 行向量 ndarray
#%% 一级标签 8 分类问题
Data = pd.read_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/FirstLevelPCA.csv",encoding='gbk')
X_dataframe = Data.iloc[:,0:-1] # 分出数据和标签 此时是DataFrame格式
y_dataframe = Data.iloc[:,-1]
X = X_dataframe.values # ndarray格式 样本数×维数
y_category = y_dataframe.values # ndarray格式
Label = LabelEncoder() # 初始化1个独热编码类
y = Label.fit_transform(y_category) # 自动生成标签
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.7,random_state=1) # 训练数据比例7成
C = 0.5 # 正则化系数,值越小泛化能力强,但是容易欠拟合,反之容易过拟合
tol = 0.0001 # 迭代阈值
models = (svm.SVC(kernel='linear', C=C,tol=tol),
          svm.SVC(kernel='rbf', gamma=0.1, C=C,tol=tol),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C,tol=tol), # 'auto' gamma= 1 / n_features.
          svm.SVC(kernel='sigmoid',C=C,tol=tol))
models = (clf.fit(X, y) for clf in models)
titles = ('线性核函数',
          '径向基核函数',
          '多项式核函数(3阶)',
          'Sigmoid核函数')
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
X0, X1 = X[:, 0], X[:, 1] # 分别是第一列和第二列
xx, yy = make_meshgrid(X0, X1)
for clf, title, ax in zip(models, titles, sub.flatten()):
    Z = plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.hsv, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.hsv, s=40, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('PCA2')
    ax.set_ylabel('PCA1')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
plt.show() # 不分训练和测试集的不同核函数的横向对比
#%%  二级标签 5 分类问题
Data = pd.read_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/SecondLevelPCA.csv",encoding='gbk')
X_dataframe = Data.iloc[:,0:-1] # 分出数据和标签 此时是DataFrame格式
y_dataframe = Data.iloc[:,-1]
X = X_dataframe.values # ndarray格式 样本数×维数
y_category = y_dataframe.values # ndarray格式
Label = LabelEncoder() # 初始化1个独热编码类
y = Label.fit_transform(y_category) # 自动生成标签
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=5) # 训练数据比例7成
C = 1.0 # 正则化系数,值越小泛化能力强,但是容易欠拟合,反之容易过拟合
tol = 0.0001 # 迭代阈值
models = (svm.SVC(kernel='linear', C=C,tol=tol),
          svm.SVC(kernel='rbf', gamma=0.1, C=C,tol=tol),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C,tol=tol), # 'auto' gamma= 1 / n_features.
          svm.SVC(kernel='sigmoid',C=C,tol=tol))
models = (clf.fit(X, y) for clf in models)
titles = ('线性核函数',
          '径向基核函数',
          '多项式核函数(3阶)',
          'Sigmoid核函数')
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.4, hspace=0.4)
X0, X1 = X[:, 0], X[:, 1] # 分别是第一列和第二列
xx, yy = make_meshgrid(X0, X1)
for clf, title, ax in zip(models, titles, sub.flatten()):

    Z = plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.hsv, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.hsv, s=40, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('PCA2')
    ax.set_ylabel('PCA1')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
plt.show() # 不分训练和测试集的不同核函数的横向对比
