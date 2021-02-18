#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : SVM最优模型参数.py
@Author : chenbei
@Date : 2020/12/27 11:32
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
from sklearn.model_selection import LeaveOneOut # 留1法, K折交叉验证中K=n(样本数)的情况
import pandas as pd
import numpy as np
from time import time
Data = pd.read_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/FirstLevelPCA.csv",encoding='gbk')
X_dataframe = Data.iloc[:,0:-1] # 分出数据和标签 此时是DataFrame格式
y_dataframe = Data.iloc[:,-1]
X = X_dataframe.values # ndarray格式 样本数×维数
y_category = y_dataframe.values # ndarray格式
Label = LabelEncoder() # 初始化1个独热编码类
y = Label.fit_transform(y_category) # 自动生成标签
loo = LeaveOneOut()
K = 0
num = 0 # 计算预测和实际相等的次数
ACU = [] # 用于存放每次预测时  当前已经预测正确的个数与当前训练次数的比值, 例如可能训练5次3次准确,训练10次中6次准确,直到最终241次有几个样本准确是真实准确率
# 由此可以得到训练过程中ACU的变化曲线
EIK = pd.DataFrame()
starttime = time()
#starttime = time()
for train_index, test_index in loo.split(X):
    train_X, train_y = X[train_index], y[train_index]
    test_X, test_y = X[test_index], y[test_index]
    idx = loo.get_n_splits(X)  # n_splits = 241
    if idx:
        K = K + 1
    clf1 = svm.SVC(kernel='linear', C=1, probability=True)
    clf1.fit(train_X , train_y)  # 训练模型
    y_pre1 = clf1.predict(test_X)
    if y_pre1 == test_y :
       num = num + 1
    acu = num / K # 当前预测正确的次数 / 训练的次数
    ACU.append(acu)
endtime = time()
Time = endtime -starttime
ACU = pd.Series(ACU)
fig, ax = plt.subplots()
ax.grid(axis='y')
ACU.plot.kde(ax=ax, linewidth=2.5,bw_method=0.3)
ACU.plot.hist(density=True,bins=12,color='c', ax=ax)
ax.set_xlabel('准确率')
ax.set_ylabel('频数')
ax.axis('tight')
labels = ["核密度估计图","直方图"]
ax.legend(labels,loc='upper left')
ax1 = ax.twinx()
ACU = ACU.values
ax1.plot(ACU,ACU,'b-v',linewidth=0.5)
meanACU = np.mean(ACU)
minACU = np.min(ACU)
ax1.plot(ACU,label='准确率')
ax1.plot([0,len(ACU)],[meanACU,meanACU],'r--',label='平均准确率')
ax1.plot([0,len(ACU)],[ACU[-1],ACU[-1]],'g--',label='真实准确率')
ax1.plot([0,len(ACU)],[minACU,minACU],'y--',label='最小准确率')
ax1.fill_betweenx(np.arange(min(ACU)-0.05,1.02,0.05),1.0,1.02,alpha=0.2,color="r",lw=2)
ax1.text(  (min(ACU)+1.0)/2 ,meanACU+0.002,"平均准确率 : "+str(round(meanACU,5)),
         verticalalignment='center',horizontalalignment='center',fontsize=14 )
ax1.text(  (min(ACU)+1.0)/2 ,ACU[-1]+0.002,"真实准确率 : "+str(round(ACU[-1],5)),
         verticalalignment='center',horizontalalignment='center',fontsize=14 )
ax1.text(  (min(ACU)+1.0)/2 ,minACU+0.002,"最小准确率 : "+str(round(minACU,5)),
         verticalalignment='center',horizontalalignment='center',fontsize=14 )
ax1.text((1.0+1.02)/2,(min(ACU)+1)/2 ,"区域不存在" ,fontsize=10.5,family='SimHei',verticalalignment='center',horizontalalignment='center')
ax1.text( (min(ACU)+1.0)/2,min(ACU)+0.01,"训练时间 : "+str(round(Time,5))+' s',
         verticalalignment='center',horizontalalignment='center',fontsize=14 )
ax1.axis('tight')
ax1.legend(('折线图','平均值'),loc='center left')
ax1.set_xlim(min(ACU),1.02)
ax1.set_ylim(min(ACU)-0.005,1.0)
fig.tight_layout()
plt.title(f'线性核Linear留1法准确率变化曲线(训练次数:{len(ACU)}次)')
plt.show()