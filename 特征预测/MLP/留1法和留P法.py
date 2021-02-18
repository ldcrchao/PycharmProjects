#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : 留1法和留P法.py
@Author : chenbei
@Date : 2020/12/23 21:44
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
from sklearn import svm
import pandas as pd
import numpy as np
from time import time
from sklearn.model_selection import LeaveOneOut # 留1法, K折交叉验证中K=n(样本数)的情况
from sklearn.model_selection import LeavePOut  # 留P法, K折交叉验证中K=n-p的情况,即剩下p个测试集
def OnetrainLeaveOne(clf,X,y,loo,p):
    K = 0
    num = 0

    sumacu = 0
    ACU = []
    for train_index, test_index in loo.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        idx = loo.get_n_splits(X)  # n_splits = 241
        pp = 0
        if idx:
            K = K + 1
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if p == 1 :
           if y_pred == y_test: # 每次交叉验证只预测1个测试集,故直接进行比较即可
               num = num + 1
           acu = num / K # 准确率 (判断当前240交叉验证次数下有多少个预测正确)
        else: # 留2法时还需要计算每次2个中有几个正确,取值只有0,0.5,1.0
            for j in range(len(y_pred)):
                if y_pred[j] == y_test[j]:
                    pp = pp + 1
            acu = pp / len(y_pred) # 0,0.5,1.0
            sumacu = sumacu + acu # 到当前k次训练类似的准确个数
            acu = sumacu / K  # 准确率
        ACU.append(acu)
    return ACU , K #返回每次训练的当前准确率和最终训练次数
def PlotACU(ACU,title,TrainingNums,Time) :
    plt.plot(ACU, 'c-p', linewidth=2, markersize=2,label='准确率')
    plt.plot([1, TrainingNums], [min(ACU), min(ACU)],'r-o', label='准确率最小值', linewidth=1)
    plt.plot([1, TrainingNums], [np.mean(ACU), np.mean(ACU)],'b-o', label='准确率平均值', linewidth=1)
    plt.fill_between(np.arange(1, TrainingNums, 1), np.mean(ACU) + np.std(ACU), np.mean(ACU) - np.std(ACU), alpha=0.1,
                     color='r')
    plt.text((1 + TrainingNums) / 2, np.mean(ACU) + 0.008, "Avarage ACU : " + str(round(np.mean(ACU), 5)),
             family="Times New Roman",horizontalalignment='center', fontsize=16)
    plt.text((1 + TrainingNums) / 2, min(ACU) + 0.005, "Min ACU : " + str(round(min(ACU), 5)), family="Times New Roman",
             fontsize=16,horizontalalignment='center')
    plt.text((1 + TrainingNums) / 2, (min(ACU) + max(ACU)) / 2, "Std ACU : " + str(round(np.std(ACU), 5)),
             family="Times New Roman",horizontalalignment='center', fontsize=16)
    plt.text((1 + TrainingNums) / 2, min(ACU) + 0.008, "Training Time : " + str(round(Time/TrainingNums, 3)) + ' s',
             verticalalignment='center', horizontalalignment='center', family='Times New Roman', fontsize=16)
    plt.title(title+f'准确率变化图')
    plt.ylabel('准确率')
    plt.xlabel('训练次数')
    plt.legend(loc='lower left')
    plt.show()
def LeaveP_Method(kernel,title,X,y,p) :
    if p == 1:
       loo = LeaveOneOut()
    else:
        loo = LeavePOut(p=p)
    clf = svm.SVC(kernel=kernel, C=1, probability=True)
    A = time()
    acu, TrainingNums = OnetrainLeaveOne(clf,X,y,loo,p) # 随着交叉验证/训练次数的变化而变化的准确率
    B = time()
    Time = B-A
    PlotACU(acu,title,TrainingNums,Time)
Data = pd.read_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/FirstLevelPCA.csv",encoding='gbk')
X_dataframe = Data.iloc[:,0:-1] # 分出数据和标签 此时是DataFrame格式
y_dataframe = Data.iloc[:,-1]
X = X_dataframe.values # ndarray格式 样本数×维数
y_category = y_dataframe.values # ndarray格式
Label = LabelEncoder() # 初始化1个独热编码类
y = Label.fit_transform(y_category) # 自动生成标签
#%% 留1法 每次只留1个,训练次数固定为交叉验证次数 ,比例也不会变每次样本数量都是240个,没有2种情况 每次训练时间基本没变化不考虑时间
LeaveP_Method('linear','SVM线性核函数留1法',X=X,y=y,p=1)
LeaveP_Method('rbf','SVM径向基核函数留1法',X=X,y=y,p=1)
#%% 留2法 每次留2个 比例也是固定的
LeaveP_Method('linear','SVM线性核函数留2法',X=X,y=y,p=2)
#%%
LeaveP_Method('rbf','SVM径向基核函数留2法',X=X,y=y,p=2)


