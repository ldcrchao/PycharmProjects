# %%
# -*- coding UTF-8 -*-
'''
@Project : MyProjects
@File : 电流后台函数测试.py
@Author : chenbei
@Date : 2021/1/27 21:09
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
import pandas as pd
import numpy as np
import math
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold  # K折交叉验证
import  os
from time import time
# 1、读取波形数据的函数
def read_current_waveform_data() :
    ''':return dataframe 1000×8'''
    path = os.getcwd() + "/MyPlatform/current_waveform_data.csv"
    waveform_data = pd.read_csv(path,encoding='gbk')
    return waveform_data

# 2、拆分波形数据得到8种波形的函数
def spilit_current_waveform_data(option):
    ''':arg : string(160,180,200,220,240,stuck,age_10,age_50)
    :return : 某个波形的series'''
    waveform_data = read_current_waveform_data()
    keys = ["160","180","200","220","age_10","age_50","stuck","240"]
    dict_waveform_data = {}
    for index , column in enumerate(waveform_data.columns) :
        dict_waveform_data.update({keys[index]:waveform_data[column]})
    return dict_waveform_data[option]
#data = spilit_waveform_data("240")

# 3、计算传入波形的数据特征
def calculate_waveform_characteristics(option):
    ''':arg : 选择哪类波形 :return 波形特征 list类型
    :internet https://blog.csdn.net/baidu_38963740/article/details/110940823'''
    option_waveform_data =  spilit_current_waveform_data(option).values # 转换成数组格式
    # 1、平均值
    waveform_mean = np.mean(option_waveform_data)
    # 2、绝对平均值
    waveform_absolute_mean = waveform_mean / len(option_waveform_data)
    # 3、均方根值 = sqrt(Σxi^2 / N)
    waveform_rms = math.sqrt(sum([x ** 2 for x in option_waveform_data]) / len(option_waveform_data ))
    # 4、方根幅值 : 所有点取绝对值再开根号 求和 求和后除长度N 之后再平方
    waveform_sra =pow (sum([ math.sqrt(abs(x)) for x in option_waveform_data]) / len(option_waveform_data),2)
    # 5、方差
    waveform_var = np.var(option_waveform_data)
    # 6、标准差
    waveform_std = np.std(option_waveform_data)
    # 7、最大值
    waveform_max = np.max(option_waveform_data)
    # 8、最小值
    waveform_min = np.min(option_waveform_data)
    # 9、偏度
    waveform_skew = np.mean((option_waveform_data - waveform_mean ) ** 3) \
                    / pow(waveform_std, 3)
    # 10、峭度
    waveform_kurt = np.mean((option_waveform_data - waveform_mean) ** 4)\
                    / pow(np.var(option_waveform_data), 2)
    # 11、峰值
    waveform_peak = max(abs(waveform_max),abs(waveform_min))
    # 12、裕度因子 = 峰值 / 方根幅值
    waveform_margin_factor = waveform_peak / waveform_sra
    # 13、波形因子 = 均方根值 / 均值
    waveform_shape_factor = waveform_rms / waveform_mean
    # 14、脉冲因子 = 峰值 / 均值
    waveform_impluse_factor = waveform_peak / waveform_mean
    # 15、峰值因子 = 峰值 / 均方根值
    waveform_crest_factor = waveform_peak / waveform_rms
    waveform_features = [round(waveform_mean,3) ,round(waveform_absolute_mean,3),round(waveform_rms,3), round(waveform_sra,3),
                         round(waveform_var, 3), round(waveform_std,3) ,round(waveform_max,3),round(waveform_min,3),
                         round(waveform_skew,3) ,round(waveform_kurt,3),round(waveform_peak,3),
                         round(waveform_margin_factor,3),round(waveform_shape_factor,3),
                         round(waveform_impluse_factor,3),round(waveform_crest_factor,3)]
    return waveform_features
# waveform_features = calculate_waveform_characteristics("220")

#  4、 导入主成分降维后的特征数据 (8分类)
def read_current_waveform_pca_features() :
    path = os.getcwd() + "/MyPlatform/current_pca_features_data.csv"
    pca_features_data =pd.read_csv(path,encoding="gbk")
    return pca_features_data
# 5、 分离pca特征数据的标签和数据
def split_current_pca_features_data_label():
    pca_features_data = read_current_waveform_pca_features()
    X_dataframe = pca_features_data.iloc[:, 0:-1]  # 分出数据和标签 此时是DataFrame格式
    y_dataframe = pca_features_data.iloc[:, -1] # 第3列是label
    X = X_dataframe.values  # ndarray格式 样本数×维数
    y_category = y_dataframe.values  # ndarray格式
    Label = LabelEncoder()  # 初始化1个独热编码类
    y = Label.fit_transform(y_category)  # 自动生成标签 不一定按顺序标 本文的数据是标签4和7互换
    return X ,y

# 6、继续分离pca特征的数据得到8种故障类型的pca用于模型预测
def spilit_current_pca_features_data(option):
    # pca的数据分离顺序是符合keys的顺序
    X, _ = split_current_pca_features_data_label()
    pca_features = [X[0:30,:] ,X[30:61, :],X[61:91, :],X[91:121, :],
                         X[121:151, :], X[151:181, :],X[181:211, :],X[211:241, :]]
    dict_pca_features_data = {}
    keys = ["160", "180", "200", "220","240", "age_10", "age_50" , "stuck"]
    for index , key in enumerate(keys) :
        dict_pca_features_data.update({key:pca_features[index]})
    return dict_pca_features_data[option]
# dataframe = spilit_pca_features_data("stuck")

# 7、返回svm的普通训练模型
def return_current_svm_model():
    X, y = split_current_pca_features_data_label()
    clf = svm.SVC(kernel='linear', C=1, probability=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
    clf.fit(X_train, y_train)
    return clf
# 8、根据选项option返回svm预测的标签
def return_current_predict_label_svm(option) :
    option_pca_features_data = spilit_current_pca_features_data(option)
    clf = return_current_svm_model()
    y_predict = clf.predict(option_pca_features_data)
    return max(list(y_predict), key=list(y_predict).count) # 返回最多的那个元素 该方法列表才有
# y_predict = return_predict_label_svm("240")
# 9、根据选项option返回svm预测的概率矩阵
def return_current_predict_proba_svm(option) :
    option_pca_features_data = spilit_current_pca_features_data(option)
    clf = return_current_svm_model()
    y_predict_proba = clf.predict_proba(option_pca_features_data)
    return y_predict_proba
# y_predict_proba = return_predict_proba_svm("240")
# 10、对返回预测的svm概率矩阵求平均
def return_current_predict_mean_proba_svm(option):
    y_predict_proba = return_current_predict_proba_svm(option)
    y_predict_proba_mean = y_predict_proba.mean(axis=0)
    return y_predict_proba_mean
#y_predict_proba_mean = return_predict_proba_svm("stuck")
# 11、返回只带交叉验证的svm训练模型 用于内层循环
def return_current_kfold_predict_proba_svm(kfold,option):
    option_pca_features_data = spilit_current_pca_features_data(option) # option对应的测试数据
    X, y = split_current_pca_features_data_label()
    state = np.random.get_state()  # 必须先打乱原来的数据集合标签集,否则初始的有序状态会导致交叉验证预测率很低
    np.random.shuffle(X)
    np.random.set_state(state)  # 保证样本和标签以相同的规律被打乱
    np.random.shuffle(y)
    Predict_proba = [] # 用于装载kfold折的概率向量 得到概率矩阵
    kf = KFold(n_splits=kfold)
    for train_index, test_index in kf.split(X): # 内层交叉训练kfold次
        train_X, train_y = X[train_index], y[train_index]
        clf = svm.SVC(kernel='linear', C=1, probability=True)
        clf.fit(train_X, train_y)  # 训练模型
        predict_proba = clf.predict_proba(option_pca_features_data) # 每一折都会生成预测概率矩阵
        predict_proba_mean = predict_proba.mean(axis=0) # 把概率矩阵先变成概率向量
        Predict_proba.append(predict_proba_mean) # 重新生成概率矩阵
    return np.array(Predict_proba).mean(axis=0) # 但是返回继续平均生成概率向量 用于后续训练次数进行平均计算
#Predict_proba =return_current_kfold_predict_proba_svm(10,"220")
# 12、返回带交叉验证的和训练次数的svm训练模型 用于外层循环
def return_current_maxiter_predict_proba_svm(maxiter,kfold,option):
    # 外层循环调用交叉验证
    Predict_proba = []  # 用于装载maxiter次后的概率向量 得到概率矩阵
    for i in range(maxiter) :
        predict_kfold_proba_mean = return_current_kfold_predict_proba_svm(kfold,option)
        Predict_proba.append(predict_kfold_proba_mean)
    return np.array(Predict_proba).mean(axis=0)
#Predict_proba = return_current_maxiter_predict_proba_svm(100,100,"220")
# 13、返回mlp的模型
def return_current_mlp_model():
    X, y = split_current_pca_features_data_label()
    clf = MLPClassifier(activation='identity', solver='lbfgs',
                        alpha=0.1, hidden_layer_sizes=(5, 2), random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.9)
    clf.fit(X_train, y_train)
    return clf

# 14、根据选项option返回mlp预测的标签
def return_current_predict_label_mlp(option) :
    option_pca_features_data = spilit_current_pca_features_data(option)
    clf = return_current_mlp_model()
    y_predict = clf.predict(option_pca_features_data)
    return max(list(y_predict), key=list(y_predict).count)

# 15、根据选项option返回预测的mlp概率矩阵
def return_current_predict_proba_mlp(option) :
    option_pca_features_data = spilit_current_pca_features_data(option) # 得到指定option的pca数据
    clf = return_current_mlp_model()
    y_predict_proba = clf.predict_proba(option_pca_features_data)
    return y_predict_proba
# y_predict_proba = return_predict_proba_mlp("240") # 对应标签7
# 16、对返回预测的mlp概率矩阵求平均
def return_current_predict_mean_proba_mlp(option):
    y_predict_proba = return_current_predict_proba_mlp(option)
    y_predict_proba_mean = y_predict_proba.mean(axis=0)
    return y_predict_proba_mean
# mean_mlp = return_current_predict_mean_proba_mlp("stuck")

# 17、返回只带交叉验证的mlp训练模型 用于内层循环
def return_current_kfold_predict_proba_mlp(kfold,option):
    option_pca_features_data = spilit_current_pca_features_data(option) # option对应的测试数据
    X, y = split_current_pca_features_data_label()
    state = np.random.get_state()  # 必须先打乱原来的数据集合标签集,否则初始的有序状态会导致交叉验证预测率很低
    np.random.shuffle(X)
    np.random.set_state(state)  # 保证样本和标签以相同的规律被打乱
    np.random.shuffle(y)
    Predict_proba = [] # 用于装载kfold折的概率向量 得到概率矩阵
    kf = KFold(n_splits=kfold)
    for train_index, test_index in kf.split(X): # 内层交叉训练kfold次
        train_X, train_y = X[train_index], y[train_index]
        clf = MLPClassifier(activation='identity', solver='lbfgs',
                        alpha=0.1, hidden_layer_sizes=(5, 2), random_state=1)
        clf.fit(train_X, train_y)  # 训练模型
        predict_proba = clf.predict_proba(option_pca_features_data) # 每一折都会生成预测概率矩阵
        predict_proba_mean = predict_proba.mean(axis=0) # 把概率矩阵先变成概率向量
        Predict_proba.append(predict_proba_mean) # 重新生成概率矩阵
    return np.array(Predict_proba).mean(axis=0) # 但是返回继续平均生成概率向量 用于后续训练次数进行平均计算
#Predict_proba =return_current_kfold_predict_proba_svm(10,"220")
# 18、返回带交叉验证的和训练次数的mlp训练模型 用于外层循环
def return_current_maxiter_predict_proba_mlp(maxiter,kfold,option):
    # 外层循环调用交叉验证
    Predict_proba = []  # 用于装载maxiter次后的概率向量 得到概率矩阵
    for i in range(maxiter) :
        predict_kfold_proba_mean = return_current_kfold_predict_proba_svm(kfold,option)
        Predict_proba.append(predict_kfold_proba_mean)
    return np.array(Predict_proba).mean(axis=0)
#Predict_Proba = return_current_maxiter_predict_proba_mlp(10,10,"240")

# 19、计算带有maxiter和kfold的模型训练耗时
def return_model_traingtime(model,maxiter,kfold,option) :
    if model=="svm" :
        starttime = time()
        svm = return_current_maxiter_predict_proba_mlp(maxiter,kfold,option)
        endtime = time()
        return svm,round(endtime-starttime,5)
    elif model == "mlp":
        starttime = time()
        mlp = return_current_maxiter_predict_proba_mlp(maxiter,kfold,option)
        endtime = time()
        return mlp,round(endtime-starttime,5)
    else:
        print("返回模型训练时间错误...")
        return False
predict_Proba , usetime = return_model_traingtime("svm",10,10,"240")

