#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : CompareMLP参数得到最优模型.py
@Author : chenbei
@Date : 2020/12/24 19:23
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
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
def MLP_NeuralNetwork(clf,X,y,trainsize=0.7) :
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=trainsize)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    numX_train = len(y_train) # 返回训练样本数量
    sum = 0
    for i in range(len(y_pred)) :
        if y_pred[i] == y_test[i] :
            sum = sum + 1
    sum = sum / len(y_pred)
    return sum ,numX_train
def get_ACU(clf,maxiter,X,y) :
    ACU = []
    num_X_train = []
    for i in range(maxiter): # 训练代数改变
        acu , num_x_train = MLP_NeuralNetwork(clf, X, y, trainsize=0.7) # 这是训练一次得到的准确率
        ACU.append(acu)
        num_X_train.append(num_x_train)
    return ACU ,num_X_train # 返回样本数量和这maxter次的acu
def get_ACU_TrainSize(clf,maxiter,trainsize,X,y) :
    '''
    与get_ACU函数区别在于transize不是固定的0.7
    :param clf: 模型
    :param maxiter: 训练次数
    :param test_size: 固定的比例
    :return: 每次训练的准确率
    '''
    ACU = []
    num_X_train = []
    for i in range(maxiter): # 训练代数改变
        acu , num_x_train = MLP_NeuralNetwork(clf, X, y, trainsize=trainsize) # 这是训练一次得到的准确率
        ACU.append(acu)
        num_X_train.append(num_x_train)
    return ACU ,num_X_train # 返回样本数量和这maxter次的acu
def PlotACU(ACU,maxiter,title1,title2) :
    plt.plot(ACU, 'c-p', linewidth=1,markersize=2, label='准确率')
    plt.plot([1, maxiter], [min(ACU), min(ACU)], 'r-o',label='准确率最小值', linewidth=1)
    plt.plot([1, maxiter], [np.mean(ACU), np.mean(ACU)], 'b-o',label='准确率平均值', linewidth=1 )
    plt.fill_between(np.arange(1, maxiter, 1), np.mean(ACU) + np.std(ACU), np.mean(ACU) - np.std(ACU), alpha=0.1,color='r')
    plt.text((1 + maxiter) / 2, np.mean(ACU) + 0.005, "准确率平均值 : " + str(round(np.mean(ACU), 5)),horizontalalignment='center',color='b', fontsize=16)
    plt.text((1 + maxiter) / 2, min(ACU) + 0.005, "准确率最小值 : " + str(round(min(ACU), 5)),color='r',horizontalalignment='center', fontsize=16)
    plt.title(f'不同{title1}MLP神经网络准确率变化图({title2})')
    plt.ylabel('准确率')
    plt.xlabel('训练次数')
    plt.legend(loc='lower left')
    plt.show()
def ChangeActivation(activation,X,y,title1,title2):
    clf = MLPClassifier(activation=activation,solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(5, 2), random_state=1)
    maxiter = 100
    ACU, _ = get_ACU(clf, maxiter, X=X, y=y)
    PlotACU(ACU,maxiter=maxiter,title1=title1,title2 =title2)
def ChangeSovler(sovler,X,y,title1,title2) :
    if sovler == 'sgd' : # 做判断是因为只有sgd方法需要设定学习率和学习方法
        clf = MLPClassifier(activation='identity', solver=sovler, alpha=1e-1, hidden_layer_sizes=(5, 2), random_state=1,
                            learning_rate='constant',learning_rate_init=0.001) # 在这里不考虑不同学习率方法,只考虑不同权值优化方法
    else:
        clf = MLPClassifier(activation='identity', solver=sovler, alpha=1e-1, hidden_layer_sizes=(5, 2), random_state=1)
    maxiter = 100
    ACU, _ = get_ACU(clf, maxiter, X=X, y=y)
    PlotACU(ACU, maxiter=maxiter, title1=title1, title2=title2)
def PlotAlpha(MinACU,MeanACU,Alphas) :
    '''
    :param MinACU: 不同正则化系数的最小值
    :param MeanACU: 不同正则化系数的平均值
    :param Alphas: 正则化系数
    :return: 关于正则化的变化趋势
    '''
    plt.semilogx(Alphas,MinACU,'r-p',label='准确率最小值',linewidth=1.5)
    plt.semilogx(Alphas,MeanACU, 'b-o', label='准确率平均值', linewidth=1.5)
    plt.legend(loc ='lower left')
    ax = plt.gca()
    locs, labels = plt.yticks()  # 刻度的位置和它使用的数值
    plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))  # 每个位置上使用新的刻度值,可以将小数刻度变为指数刻度
    for tick in ax.xaxis.get_major_ticks():  # 获取图片坐标轴的主刻度,循环设置格式 解决指数坐标不能显示负号的问题
        tick.label1.set_fontproperties('stixgeneral')
    plt.xlabel('正则化系数alpha')
    plt.ylabel('准确率')
    plt.title('不同正则化系数MLP神经网络平均准确率变化图')
    plt.ylim(0,max(max(MinACU),max(MeanACU)))
    plt.show()
def ChangeAlpha(alpha,X,y,title1,title2) :
    clf = MLPClassifier(activation='identity', solver='lbfgs', alpha=alpha, hidden_layer_sizes=(5, 2), random_state=1)
    maxiter = 100
    ACU, _ = get_ACU(clf, maxiter, X=X, y=y)
    PlotACU(ACU, maxiter=maxiter, title1=title1, title2=title2)
    return ACU
def ChangeLearingRate(learning_rate,X,y,title1,title2) :
    if learning_rate == 'constant' :
       clf = MLPClassifier(activation='identity', solver='sgd',learning_rate=learning_rate,learning_rate_init=0.001,
                        alpha=0.1, hidden_layer_sizes=(5, 2), random_state=1)
    else:
        clf = MLPClassifier(activation='identity', solver='sgd',learning_rate=learning_rate,learning_rate_init=0.001,
                        alpha=0.1,power_t=0.5, hidden_layer_sizes=(5, 2), random_state=1)
    maxiter = 100
    ACU, _ = get_ACU(clf, maxiter, X=X, y=y)
    PlotACU(ACU, maxiter=maxiter, title1=title1, title2=title2)
def ChangeConstantLearningRateInit(learning_rate_init,X,y,title1,title2):
    clf = MLPClassifier(activation='identity', solver='sgd', learning_rate='constant', learning_rate_init=learning_rate_init,
                        alpha=0.1, hidden_layer_sizes=(5, 2), random_state=1)
    maxiter = 100
    ACU, _ = get_ACU(clf, maxiter, X=X, y=y)
    PlotACU(ACU, maxiter=maxiter, title1=title1, title2=title2)
    return ACU
def PlotConstantLearningRateInit(MinACU,MeanACU,Learning_rate_inits) :
    plt.semilogx(Learning_rate_inits,MinACU,'r-p',label='准确率最小值',linewidth=1.5)
    plt.semilogx(Learning_rate_inits,MeanACU, 'b-o', label='准确率平均值', linewidth=1.5)
    plt.legend(loc ='lower left')
    ax = plt.gca()
    locs, labels = plt.yticks()  # 刻度的位置和它使用的数值
    plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))  # 每个位置上使用新的刻度值,可以将小数刻度变为指数刻度
    for tick in ax.xaxis.get_major_ticks():  # 获取图片坐标轴的主刻度,循环设置格式 解决指数坐标不能显示负号的问题
        tick.label1.set_fontproperties('stixgeneral')
    plt.xlabel('学习率')
    plt.ylabel('准确率')
    plt.title('不同学习率MLP神经网络平均准确率变化图(SGD-Constant)')
    plt.ylim(0,max(max(MinACU),max(MeanACU)))
    plt.show()
def ChangeInvscalingLearningRateInit(learning_rate_init,X,y,title1,title2):
    clf = MLPClassifier(activation='identity', solver='sgd', learning_rate='invscaling', learning_rate_init=learning_rate_init,
                        alpha=0.1, hidden_layer_sizes=(5, 2), random_state=1)
    maxiter = 100
    ACU, _ = get_ACU(clf, maxiter, X=X, y=y)
    PlotACU(ACU, maxiter=maxiter, title1=title1, title2=title2)
    return ACU
def PlotInvscalingLearningRateInit(MinACU,MeanACU,Learning_rate_inits) :
    plt.semilogx(Learning_rate_inits,MinACU,'r-p',label='准确率最小值',linewidth=1.5)
    plt.semilogx(Learning_rate_inits,MeanACU, 'b-o', label='准确率平均值', linewidth=1.5)
    plt.legend(loc ='lower left')
    ax = plt.gca()
    locs, labels = plt.yticks()  # 刻度的位置和它使用的数值
    plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))  # 每个位置上使用新的刻度值,可以将小数刻度变为指数刻度
    for tick in ax.xaxis.get_major_ticks():  # 获取图片坐标轴的主刻度,循环设置格式 解决指数坐标不能显示负号的问题
        tick.label1.set_fontproperties('stixgeneral')
    plt.xlabel('学习率')
    plt.ylabel('准确率')
    plt.title('不同学习率MLP神经网络平均准确率变化图(SGD-Invscaling)')
    plt.ylim(0,max(max(MinACU),max(MeanACU)))
    plt.show()
def ChangeInvscalingPowert(power_t,X,y,title1,title2):
    clf = MLPClassifier(activation='identity', solver='sgd', learning_rate='invscaling', learning_rate_init=0.1,
                        alpha=0.1, hidden_layer_sizes=(5, 2), random_state=1,power_t= power_t)
    maxiter = 100
    ACU, _ = get_ACU(clf, maxiter, X=X, y=y)
    PlotACU(ACU, maxiter=maxiter, title1=title1, title2=title2)
    return ACU
def PlotInvscalingPowert(MinACU,MeanACU,Power_t) :
    plt.semilogx(Power_t,MinACU,'r-p',label='准确率最小值',linewidth=1.5)
    plt.semilogx(Power_t,MeanACU, 'b-o', label='准确率平均值', linewidth=1.5)
    plt.legend(loc ='lower left')
    ax = plt.gca()
    locs, labels = plt.yticks()  # 刻度的位置和它使用的数值
    plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))  # 每个位置上使用新的刻度值,可以将小数刻度变为指数刻度
    for tick in ax.xaxis.get_major_ticks():  # 获取图片坐标轴的主刻度,循环设置格式 解决指数坐标不能显示负号的问题
        tick.label1.set_fontproperties('stixgeneral')
    plt.xlabel('逆比例学习率指数')
    plt.ylabel('准确率')
    plt.title('不同学习率MLP神经网络平均准确率变化图(SGD-Invscaling-Power_t)')
    plt.ylim(0,max(max(MinACU),max(MeanACU)))
    plt.show()
def PlotTrainingSampleRatio(ACU_X,ACU,time,maxiter) :
    fig ,ax = plt.subplots() # 传入的为训练集数量和对应的准确率
    ax.plot(ACU_X,ACU, 'g-v', linewidth=2,markersize=2, label='平均准确率')
    ax.plot([1, max(ACU_X)],[min(ACU), min(ACU)],'r-o', label='平均准确率最小值', linewidth=1)
    ax.plot([1, max(ACU_X)],[np.mean(ACU), np.mean(ACU)], 'b-o' ,label='平均准确率平均值', linewidth=1)
    ax.fill_between(np.arange(1, max(ACU_X), 1),
                    np.mean(ACU) + np.std(ACU), np.mean(ACU) - np.std(ACU),
                    alpha=0.1,color='r')
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
def ChangeTrainingSampleRatio(X,y):
    trainsizes = np.arange(0.1, 1.0, 0.1)# 最优参数下划分训练集比例
    clf1 = MLPClassifier(activation= 'identity',solver='lbfgs', alpha=0.1,hidden_layer_sizes=(5, 2), random_state=1)
    maxiter = 100
    ACUmean = []
    ACU_X = []
    Time = []
    for trainsize in trainsizes:  # 不同训练样本数
        starttime = time()
        # 对于不同训练集比例的需要返回样本数,用于绘图
        acu, acu_x = get_ACU_TrainSize(clf1, maxiter, trainsize, X=X, y=y)  # 某一个训练比例,迭代maxiter次 得到相应的所有acu
        acu_mean = np.mean(acu)  # 找到每个训练比例下迭代maiter次的准确率平均值
        ACUmean.append(acu_mean)
        acu_x_mean = np.mean(acu_x) # 100个 0.9比例 取平均或者取第1个都一样
        ACU_X.append(acu_x_mean)  # 每次训练的数量
        endtime = time()
        consumetime = endtime - starttime
        Time.append(consumetime)
    Time = np.array(Time) / maxiter  # 归算到每一次花费的时间
    PlotTrainingSampleRatio(ACU_X, ACUmean, Time, maxiter=maxiter)
Data = pd.read_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/FirstLevelPCA.csv",encoding='gbk')
X_dataframe = Data.iloc[:,0:-1] # 分出数据和标签 此时是DataFrame格式
y_dataframe = Data.iloc[:,-1]
X = X_dataframe.values # ndarray格式 样本数×维数
y_category = y_dataframe.values # ndarray格式
Label = LabelEncoder() # 初始化1个独热编码类
y = Label.fit_transform(y_category) # 自动生成标签
#MLP神经网络
'''
选定的默认参数 : 训练次数100、测试集比例0.3、
1、不同激活函数 activation default ='identity' ['identity', 'logistic', 'tanh', 'relu']
2、不同权值优化方法solver default='lbfgs' ['lbfgs', 'sgd', 'adam']
3、不同正则化系数lpha default=0.1  [0.00001,0.0001,0.001,0.01,0.1]
4、不同学习率方法learning_rate(仅在solver='sgd') default = 'constant'
4.1、'constant' ->  'learning_rate_init' : default=0.001        [0.001,0.01,0.1]
4.2、'invscaling' -> learning_rate_init / pow(t, power_t) power_t : default=0.5(不变)
'''
#%% 其他参数不变,激活函数进行改变
ChangeActivation('identity',X,y,'激活函数','线性激活函数f(x)=x')
ChangeActivation('logistic',X,y,'激活函数','S型激活函数f(x)=1/(1+exp(-x))')
ChangeActivation('tanh',X,y,'激活函数','双曲正切激活函数f(x)=tanh(x)')
ChangeActivation('relu',X,y,'激活函数','校正线性单位函数f(x)=max(0,x)')
#%% 不同权值优化方法
ChangeSovler('lbfgs',X,y,'权值优化方法','拟牛顿法优化lbfgs') # 小样本更好
ChangeSovler('adam',X,y,'权值优化方法','Kingma随机梯度优化adam') # 适合数千以上的样本
ChangeSovler('sgd',X,y,'权值优化方法','随机梯度下降sgd')
#%% 不同正则化系数 (旋定lbfgs , identity)
Alphas = [1e-5,1e-4,1e-3,1e-2,1e-1,0.5,1,5,10,20,50,80,100]
labels = ['alpha=0.00001','alpha=0.0001','alpha=0.001','alpha=0.01','alpha=0.1','alpha=0.5','alpha=1',
          'alpha=5','alpha=10','alpha=20','alpha=50','alpha=80','alpha=100']
MinACU = []
MeanACU = []
for i in range(len(Alphas))  :
    ACU = ChangeAlpha(Alphas[i],X,y,'正则化系数',labels[i])
    minACU = min(ACU)
    meanACU = np.mean(ACU)
    MinACU.append(minACU)
    MeanACU.append(meanACU) # 得到不同正则化系数的平均准确率和最小准确率
PlotAlpha(MinACU=MinACU,MeanACU=MeanACU,Alphas=Alphas) # 正则化系数曲线
#%% 特别的在solver='sgd'时比较不同学习方法
learning_rates = ['constant','invscaling']
labels = ['固定学习率方法constant','反比例缩放指数方法invscaling']
for i in range(len(learning_rates)) :
    ChangeLearingRate(learning_rates[i],X,y,'学习方法',labels[i])
#%% 学习方法固定 , sgd-constant 不同学习率
Learning_rate_inits = [1e-5,1e-4,1e-3,1e-2,1e-1,0.5,1,5,10,20,50,80,100]
labels = ['学习率=0.00001','学习率=0.0001','学习率=0.001','学习率=0.01','学习率=0.1','学习率=0.5','学习率=1',
          '学习率=5','学习率=10','学习率=20','学习率=50','学习率=80','学习率=100']
MinACU = []
MeanACU = []
for i in range(len(Learning_rate_inits))  :
    ACU = ChangeConstantLearningRateInit(Learning_rate_inits[i],X,y,'学习率',labels[i])
    minACU = min(ACU)
    meanACU = np.mean(ACU)
    MinACU.append(minACU)
    MeanACU.append(meanACU)
PlotConstantLearningRateInit(MinACU=MinACU,MeanACU=MeanACU,Learning_rate_inits=Learning_rate_inits)
#%% 学习方法固定 , sgd-invscaling 不同学习率
Learning_rate_inits = [1e-5,1e-4,1e-3,1e-2,1e-1,0.5,1,5,10,20,50,80,100]
labels = ['学习率=0.00001','学习率=0.0001','学习率=0.001','学习率=0.01','学习率=0.1','学习率=0.5','学习率=1',
          '学习率=5','学习率=10','学习率=20','学习率=50','学习率=80','学习率=100']
MinACU = []
MeanACU = []
for i in range(len(Learning_rate_inits))  :
    ACU = ChangeInvscalingLearningRateInit(Learning_rate_inits[i],X,y,'学习率',labels[i])
    minACU = min(ACU)
    meanACU = np.mean(ACU)
    MinACU.append(minACU)
    MeanACU.append(meanACU)
PlotInvscalingLearningRateInit(MinACU=MinACU,MeanACU=MeanACU,Learning_rate_inits=Learning_rate_inits)
#%% 学习方法固定 , sgd-invscaling-power_t 不同逆比例学习率指数
power_ts = [1e-5,1e-4,1e-3,1e-2,1e-1,0.5,1,5,10,20,50,80]
labels = ['逆比例学习率指数=0.00001','逆比例学习率指数=0.0001','逆比例学习率指数=0.001','逆比例学习率指数=0.01','逆比例学习率指数=0.1','逆比例学习率指数=0.5','逆比例学习率指数=1',
          '逆比例学习率指数=5','逆比例学习率指数=10','逆比例学习率指数=20','逆比例学习率指数=50','逆比例学习率指数=80']
MinACU = []
MeanACU = []
for i in range(len(power_ts))  :
    ACU = ChangeInvscalingPowert(power_ts[i],X,y,'学习率',labels[i])
    minACU = min(ACU)
    meanACU = np.mean(ACU)
    MinACU.append(minACU)
    MeanACU.append(meanACU)
PlotInvscalingPowert(MinACU=MinACU,MeanACU=MeanACU,Power_t=power_ts)
#%% 上述的学习率、学习方法、激活函数、权值优化方法、逆学习率指数 都是计算随着训练次数的变化而变化的准确率 (固定训练比例0.3)
# 现在还需要考虑在上述试验得到的最优参数下，即activation= 'identity',solver='lbfgs', alpha=0.1 时的不同训练样本数比例(0.1,0.2,..,0.9) 下的准确率
ChangeTrainingSampleRatio(X,y) # 可以同时观察训练时间和准确率
#%%
