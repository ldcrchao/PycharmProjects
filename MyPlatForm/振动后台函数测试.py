#%%
# -*- coding UTF-8 -*-
'''
@Project : MyProjects
@File : 振动后台函数测试.py
@Author : chenbei
@Date : 2021/3/14 21:24
'''
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl

plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置字体风格,必须在前然后设置显示中文
mpl.rcParams['font.size'] = 10.5  # 图片字体大小
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文的命令
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号的命令
mpl.rcParams['agg.path.chunksize'] = 10000
plt.rcParams['figure.figsize'] = (7.8, 3.8)  # 设置figure_size尺寸
plt.rcParams['savefig.dpi'] = 600  # 图片像素
plt.rcParams['figure.dpi'] = 600  # 分辨率
from matplotlib.font_manager import FontProperties
import  os
import math
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold  # K折交叉验证
from time import time
font_set = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=10.5)
import numpy as np
import pandas as pd
# 1、读取振动原始波形
def  read_vibration_waveform_data():
    path = os.getcwd() + "/MyPlatform/vibration_waveform_data.csv"
    waveform_data = pd.read_csv(path, encoding='gbk')
    return waveform_data
# waveform_data = read_vibration_waveform_data()
# 2、分离振动原始波形
def spilit_vibration_waveform_data(option):
    ''''''
    # "正常合闸", "正常分闸", "虚假合闸", "分闸不彻底",
    #                  "储能弹簧卡涩","操作机构卡涩","合闸螺栓松动", "分闸螺栓松动"
    waveform_data = read_vibration_waveform_data()
    keys = ["正常合闸","正常分闸", "虚假合闸", "分闸不彻底",
            "储能弹簧卡涩","操作机构卡涩","合闸螺栓松动", "分闸螺栓松动"]
    dict_waveform_data = {}
    for index, column in enumerate(waveform_data.columns):
        dict_waveform_data.update({keys[index]: waveform_data[column]})
    # 将所有故障波形存放在字典中，并通过option返回指定的波形
    option_waveform = dict_waveform_data[option]
    option_waveform.dropna(inplace=True)
    return option_waveform
#hezha_normal_data = spilit_vibration_waveform_data("正常分闸")
# 3、读取振动pca特征
def read_vibration_waveform_pca_features():
    path = os.getcwd() + "/MyPlatform/vibration_pca_features_data.csv"
    pca_features_data = pd.read_csv(path, encoding="gbk")
    return pca_features_data  # 241×2
#pca_features_data = read_vibration_waveform_pca_features()
# 4、分离振动标签和pca特征
def split_vibration_pca_features_data_label():
    pca_features_data = read_vibration_waveform_pca_features()
    X_dataframe = pca_features_data.iloc[:, 0:-1]  # 分出数据和标签 此时是DataFrame格式
    y_dataframe = pca_features_data.iloc[:, -1]  # 第3列是label
    X = X_dataframe.values  # ndarray格式 样本数×维数
    y_category = y_dataframe.values  # ndarray格式
    Label = LabelEncoder()  # 初始化1个独热编码类
    y = Label.fit_transform(y_category)  # 自动生成标签 不一定按顺序标 本文的数据是标签4和7互换
    return X, y
X, y = split_vibration_pca_features_data_label()
# 5、根据option读取某个故障类型的分离的振动标签和pca特征
def spilit_vibration_pca_features_data(option):
    # pca的数据分离顺序是符合keys的顺序
    X, _ = split_vibration_pca_features_data_label()
    pca_features = [X[0:11, :], X[11:21, :], X[21:32, :], X[32:43, :],
                    X[43:54, :], X[54:64, :], X[64:73, :], X[73:84, :]]
    dict_pca_features_data = {}
    keys = ["正常合闸", "正常分闸", "分闸不彻底", "虚假合闸", "储能弹簧卡涩",
            "操作机构卡涩", "合闸螺栓松动", "分闸螺栓松动"]
    for index, key in enumerate(keys):
        dict_pca_features_data.update({key: pca_features[index]})
    return dict_pca_features_data[option]
#xx = spilit_vibration_pca_features_data("虚假合闸")
# 6、根据振动pca数据和label得到svm模型
def return_vibration_svm_model():
    X, y = split_vibration_pca_features_data_label()
    clf = svm.SVC(kernel='linear', C=1, probability=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.9)
    clf.fit(X_train, y_train)
    return clf
# 7、根据返回的振动svm模型预测其标签(用不上)
def return_vibration_predict_label_svm( option):
    option_pca_features_data = spilit_vibration_pca_features_data(
        option)
    clf = return_vibration_svm_model()
    y_predict = clf.predict(option_pca_features_data)
    return max(list(y_predict), key=list(
        y_predict).count)  # 返回最多的那个元素 该方法列表才有
#xx = return_vibration_predict_label_svm("虚假合闸")
# 8、根据选项option返回振动信号的svm预测的概率矩阵
def return_vibration_predict_proba_svm(option):
    option_pca_features_data = spilit_vibration_pca_features_data(
        option)
    clf = return_vibration_svm_model()
    y_predict_proba = clf.predict_proba(option_pca_features_data)
    return y_predict_proba
# y_predict_proba = return_vibration_predict_proba_svm("虚假合闸") #得到11×8的概率矩阵 还需要求平均
# 9、对返回预测的振动信号的svm概率矩阵求平均
def return_vibration_predict_mean_proba_svm(option):
    y_predict_proba = return_vibration_predict_proba_svm(option)
    y_predict_proba_mean = y_predict_proba.mean(axis=0)
    return y_predict_proba_mean
#y_predict_proba_mean = return_vibration_predict_mean_proba_svm("虚假合闸")
# 10、返回只带交叉验证的振动信号的svm训练模型 用于内层循环
def return_vibration_kfold_predict_proba_svm(kfold, option):
    option_pca_features_data = spilit_vibration_pca_features_data(
        option)  # option对应的测试数据
    X, y = split_vibration_pca_features_data_label()
    state = np.random.get_state()  # 必须先打乱原来的数据集合标签集,否则初始的有序状态会导致交叉验证预测率很低
    np.random.shuffle(X)
    np.random.set_state(state)  # 保证样本和标签以相同的规律被打乱
    np.random.shuffle(y)
    Predict_proba = []  # 用于装载kfold折的概率向量 得到概率矩阵
    kf = KFold(n_splits=kfold)
    for train_index, test_index in kf.split(X):  # 内层交叉训练kfold次
        train_X, train_y = X[train_index], y[train_index]
        clf = svm.SVC(kernel='linear', C=1, probability=True)
        clf.fit(train_X, train_y)  # 训练模型
        predict_proba = clf.predict_proba(
            option_pca_features_data)  # 每一折都会生成预测概率矩阵
        predict_proba_mean = predict_proba.mean(axis=0)  # 把概率矩阵先变成概率向量
        Predict_proba.append(predict_proba_mean)  # 重新生成概率矩阵
    # print(np.array(Predict_proba).mean(axis=0))
    return np.array(Predict_proba).mean(
        axis=0)  # 但是返回继续平均生成概率向量 用于后续训练次数进行平均计算
# xx  = return_vibration_kfold_predict_proba_svm(8,"虚假合闸")
# 11、带有maxiter的svm的振动信号的普通训练模型 用于 交叉验证次数=0和1时调用 返回
def return_vibration_maxiter_predict_proba_svm(option):
    # 返回振动信号的训练次数平均概率
    option_pca_features_data = spilit_vibration_pca_features_data(
        option)  # option对应的测试数据
    X, y = split_vibration_pca_features_data_label()
    state = np.random.get_state()  # 必须先打乱原来的数据集合标签集,否则初始的有序状态会导致交叉验证预测率很低
    np.random.shuffle(X)
    np.random.set_state(state)  # 保证样本和标签以相同的规律被打乱
    np.random.shuffle(y)
    clf = svm.SVC(kernel='linear', C=1, probability=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.9)
    clf.fit(X_train, y_train)
    predict_proba = clf.predict_proba(option_pca_features_data)
    predict_proba_mean = predict_proba.mean(axis=0)  # 把概率矩阵先变成概率向量
    return list(predict_proba_mean)
#xx = return_vibration_maxiter_predict_proba_svm("虚假合闸")
# 12、返回振动信号的mlp的模型
def return_vibration_mlp_model():
    X, y = split_vibration_pca_features_data_label()
    clf = MLPClassifier(
        activation='identity',
        solver='lbfgs',
        alpha=0.1,
        hidden_layer_sizes=(
            5,
            2),
        random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.9)
    clf.fit(X_train, y_train)
    return clf
# 13、根据选项option返回振动信号的mlp预测的标签(用不上)
def return_vibration_predict_label_mlp(option):
    option_pca_features_data = spilit_vibration_pca_features_data(
        option)
    clf = return_vibration_mlp_model()
    y_predict = clf.predict(option_pca_features_data)
    return max(list(y_predict), key=list(y_predict).count)
#xx = return_vibration_predict_label_mlp("虚假合闸")
# 14、根据选项option返回振动信号的预测的mlp概率矩阵
def return_vibration_predict_proba_mlp(option):
    option_pca_features_data = spilit_vibration_pca_features_data(
        option)  # 得到指定option的pca数据
    clf = return_vibration_mlp_model()
    y_predict_proba = clf.predict_proba(option_pca_features_data)
    return y_predict_proba
# 15、对返回预测的振动信号的mlp概率矩阵求平均
def return_vibration_predict_mean_proba_mlp(option):
    y_predict_proba = return_vibration_predict_proba_mlp(option)
    y_predict_proba_mean = y_predict_proba.mean(axis=0)
    return y_predict_proba_mean
#xx = return_vibration_predict_mean_proba_mlp("虚假合闸")
# 16、返回只带交叉验证的振动信号的mlp训练模型 用于内层循环
def return_vibration_kfold_predict_proba_mlp(kfold, option):
    option_pca_features_data = spilit_vibration_pca_features_data(
        option)  # option对应的测试数据
    X, y = split_vibration_pca_features_data_label()
    state = np.random.get_state()  # 必须先打乱原来的数据集合标签集,否则初始的有序状态会导致交叉验证预测率很低
    np.random.shuffle(X)
    np.random.set_state(state)  # 保证样本和标签以相同的规律被打乱
    np.random.shuffle(y)
    Predict_proba = []  # 用于装载kfold折的概率向量 得到概率矩阵
    kf = KFold(n_splits=kfold)
    for train_index, test_index in kf.split(X):  # 内层交叉训练kfold次
        train_X, train_y = X[train_index], y[train_index]
        clf = MLPClassifier(
            activation='identity',
            solver='lbfgs',
            alpha=0.1,
            hidden_layer_sizes=(
                5,
                2),
            random_state=1)
        clf.fit(train_X, train_y)  # 训练模型
        predict_proba = clf.predict_proba(
            option_pca_features_data)  # 每一折都会生成预测概率矩阵
        predict_proba_mean = predict_proba.mean(axis=0)  # 把概率矩阵先变成概率向量
        Predict_proba.append(predict_proba_mean)  # 重新生成概率矩阵
    return np.array(Predict_proba).mean(
        axis=0)  # 但是返回继续平均生成概率向量 用于后续训练次数进行平均计算
# xx = return_vibration_kfold_predict_proba_mlp(8,"虚假合闸")
# 17、带有maxiter的mlp的振动信号的普通训练模型 用于 交叉验证次数=0和1时调用 返回
def return_vibration_maxiter_predict_proba_mlp( option):
    option_pca_features_data = spilit_vibration_pca_features_data(
        option)  # option对应的测试数据
    X, y = split_vibration_pca_features_data_label()
    state = np.random.get_state()  # 必须先打乱原来的数据集合标签集,否则初始的有序状态会导致交叉验证预测率很低
    np.random.shuffle(X)
    np.random.set_state(state)  # 保证样本和标签以相同的规律被打乱
    np.random.shuffle(y)
    # Predict_proba = []  # 用于装载留出法 得到的概率矩阵
    clf = MLPClassifier(
        activation='identity',
        solver='lbfgs',
        alpha=0.1,
        hidden_layer_sizes=(
            5,
            2),
        random_state=1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.9)
    clf.fit(X_train, y_train)
    predict_proba = clf.predict_proba(option_pca_features_data)
    predict_proba_mean = predict_proba.mean(axis=0)  # 把概率矩阵先变成概率向量
    # Predict_proba.append(predict_proba_mean)  # 重新生成概率矩阵
    return list(predict_proba_mean)  # 列表
#xx = return_vibration_maxiter_predict_proba_mlp("虚假合闸")