# %%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : SVM-linear-8.py
@Author : chenbei
@Date : 2020/12/17 18:54
'''
from sklearn.model_selection import ShuffleSplit  # ShuffleSplit方法，可以随机的把数据打乱，然后分为训练集和测试集
from time import time
import numpy as np
import pandas as pd
from sklearn.kernel_approximation import (RBFSampler, Nystroem)  # 内核近似
from sklearn.model_selection import permutation_test_score  # 模型评估 : 通过排序评估交叉验证的得分重要性
from sklearn.model_selection import validation_curve  # 模型评估 : 不同内核系数gamma
from sklearn.model_selection import learning_curve  # 模型评估 : 学习率曲线
from sklearn.model_selection import cross_val_predict  # 模型评估 : 交叉验证估计
from sklearn.model_selection import cross_val_score  # 模型评估 : 不同正则化系数C
from sklearn.model_selection import RepeatedKFold  # P次K折交叉验证
from sklearn.model_selection import LeavePOut  # 留P法, K折交叉验证中K=n-p的情况,即剩下p个测试集
from sklearn.model_selection import LeaveOneOut  # 留1法, K折交叉验证中K=n(样本数)的情况
from sklearn.model_selection import KFold  # K折交叉验证
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  # 留出法的分割方式
from sklearn import pipeline
from sklearn import svm
from matplotlib.font_manager import FontProperties
from matplotlib.pylab import mpl
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置字体风格,必须在前然后设置显示中文
#mpl.rcParams.update({'text.usetex': False,'font.family': 'stixgeneral','mathtext.fontset': 'stix'})
mpl.rcParams['font.size'] = 10.5  # 图片字体大小
mpl.rcParams['font.sans-serif'] = ['SimHei', 'SongTi']  # 显示中文的命令
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号的命令
#mpl.rcParams.update({'text.usetex': False,'font.family': 'stixgeneral','mathtext.fontset': 'stix',})
# plt.rcParams['figure.figsize'] = (7.8,3.8) # 设置figure_size尺寸
plt.rcParams['image.interpolation'] = 'nearest'  # 设置 interpolation style
# plt.rcParams['image.cmap'] = 'white' # 设置 颜色 style
plt.rcParams['savefig.dpi'] = 600  # 图片像素
plt.rcParams['figure.dpi'] = 400  # 分辨率
font_set = FontProperties(
    fname=r"C:\Windows\Fonts\simsun.ttc",
    size=10.5)  # matplotlib内无中文字节码，需要自行手动添加
# 它还有一个好处是可以通过random_state这个种子来重现我们的分配方式，如果没有指定，那么每次都是随机的
# RBFSampler通过其傅里叶变换的蒙特卡洛近似来近似RBF内核的特征图 ;  Nystroem使用训练数据的子集近似核图,使用数据的子集为任意内核构造一个近似特征图
# import seaborn as sns
#import random


def plt_learning_curve(estimator, title, X, y, ylim=None,
                       cv=None, train_size=np.linspace(.1, 1.0, 5)):
    '''
    :param estimator: 计算指定的学习模型estimator在不同大小的训练集经过交叉验证后的训练的分和测试得分
    :param title: 标题
    :param X: 特征矩阵 : 样本数×维数
    :param y: 标签 : 行向量
    :param ylim: y轴限制区间
    :param cv: 交叉验证次数(默认3-fold), int指定"折"的数量 , 一个产生train/test划分的迭代器对象
    :param train_size: 指定训练集子集的比例,默认np.linspace(.1,1.0,5)
    :return: 不同训练集比例的学习率曲线
    '''
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("训练集样本个数")
    plt.ylabel("得分")
    train_sizes, train_scores, test_scores, fittime, scoretime = learning_curve(
        estimator, X, y, cv=cv, n_jobs=1, train_sizes=train_size, return_times=True)
    # print(train_scores)
    # 返回已用于生成学习曲线的训练示例数、训练集得分、测试集得分、拟合时间、得分时间
    # cv规定了每次交叉验证的次数和比例
    # 返回的训练测试集得分矩阵为n_ticks × n_folds , 即列数取决于n_splits ,行数取决于train_size的个数
    # 表示训练子集的比例依次为0.1,0.109,0.118....,1时,每次交叉验证的得分
    # 例如cv中已经确定test_size为0.2,那么应当有0.8*241=192个样本作为训练集
    # 这192个训练样本中的0.1即只有19个样本真正为训练集,对(241-192=49个测试样本进行预测),进行100次交叉验证
    # 直到最后一次,192）0.9909个样本全部作为训练集对测试样本交叉验证
    # 也就是说train_size先控制子训练集的比例,test_size再控制最终的测试集
    # 按列平均将矩阵变为向量
    train_scores_mean = np.mean(train_scores, axis=1)  # 每一次交叉验证训练集得分的平均值
    train_scores_std = np.std(train_scores, axis=1)  # 每一次交叉验证训练集得分的标准差
    test_scores_mean = np.mean(test_scores, axis=1)  # 每一次交叉验证测试集得分的平均值
    test_scores_std = np.std(test_scores, axis=1)  # 每一次交叉验证测试集得分的标准差
    plt.grid()
    # 两个函数之间的区域用黄色填充
    #plt.fill_between(x, y1, y2, facecolor="yellow")
    '''为了画出带状区域,使用训练得分平均值减去或加上训练得分标准差,这两条曲线之间填充颜色'''
    # 训练集上下包络线
    plt.fill_between(
        train_sizes,
        train_scores_mean -
        train_scores_std,
        train_scores_mean +
        train_scores_std,
        alpha=0.1,
        color="r")
    # 测试集上下包络线
    plt.fill_between(
        train_sizes,
        test_scores_mean -
        test_scores_std,
        test_scores_mean +
        test_scores_std,
        alpha=0.1,
        color="g")
    # 测试集/训练集 主曲线
    plt.plot(
        train_sizes,
        train_scores_mean,
        "o-",
        color="r",
        label="交叉验证-训练集得分")  # 图例可以使用label控制
    plt.plot(
        train_sizes,
        test_scores_mean,
        "o-",
        color="g",
        label="交叉验证-测试集得分")
    plt.legend(loc="best")
    plt.axis('tight')
    plt.show()
    return plt


def EvaluationIndex(
        X_test,
        y_test,
        y_pre,
        clf,
        confusion_matrix=False,
        title=None):
    '''评价指标'''
    # 1、准确度 ACU
    # ACU = clf.score(X_test,y_test) # 返回给定测试数据和标签上的平均准确度
    ACU = metrics.accuracy_score(y_test, y_pre)  # 直接引用方法accuracy_score是相同的
    print('平均准确率ACU为:', ACU)
    # print('准确率得分为:',ACU_Score)
    # 2、精准率 PRECISON
    Precison_Score = metrics.precision_score(
        y_test, y_pre, average='weighted')  # 精准率
    print('精准率Precison为:', Precison_Score)
    # 3、召回率 RECALL
    Recall = metrics.recall_score(y_test, y_pre, average='weighted')
    print('召回率Recall为:', Recall)
    # 4、F1 得分
    F1 = metrics.f1_score(y_test, y_pre, average='weighted')
    print('F1得分为:', F1)
    # 以前的参数输入值是真实标签和预测标签,除了平均准确率外对于多类问题必须指定avarage的值
    # 5、混淆矩阵
    CM = metrics.confusion_matrix(y_test, y_pre)
    # 6、绘制混淆矩阵
    if confusion_matrix:
        plt.figure(figsize=(4.4, 3.8))
        metrics.plot_confusion_matrix(clf, X_test, y_test)
        plt.xlabel('预测标签', fontproperties=font_set)
        # plt.ylabel('实际标签',fontsize=10.5,fontname='宋体') # 会出现乱码
        plt.ylabel('实际标签', fontproperties=font_set)
        plt.title(f'{title}的混淆矩阵', fontproperties=font_set)
        plt.show()
    # 7、最大误差
    #max_error = metrics.max_error(y_test,clf.predict(X_test))
    # print('预测最大误差为:',max_error) = 7
    # 8、方均根误差
    RMS = metrics.mean_squared_error(y_test, y_pre)
    print('方均根误差RMS为:', RMS)
    # 9、显示主要的指标报告
    # report = metrics.classification_report (y_test,y_pre ,output_dict=True) # 返回字典
    # 10、汉明损失是被错误预测的标签分数
    hammingloss = metrics.hamming_loss(y_test, y_pre)
    print('Hamming损失为:', hammingloss)
    # 11、多重混淆矩阵
    # MULCM = metrics.multilabel_confusion_matrix(y_test,clf.predict(X_test))
    # Matthews相关系数
    matthews = metrics.matthews_corrcoef(y_test, y_pre)
    print('Matthews相关系数为:', matthews)
    # 12、R2系数
    R2 = metrics.r2_score(y_test, y_pre)
    print('R2系数为:', R2)
    # 13、平衡精度BAS
    BAS = metrics.balanced_accuracy_score(y_test, y_pre)
    print('平衡精度BAS为:', BAS)
    # 14、精度和召回率的加权谐波平均值F_beta
    F_beta = metrics.fbeta_score(
        y_test,
        y_pre,
        beta=0.5,
        average='weighted')  # beta 确定组合分数中的召回权重
    print('加权谐波平均值F_beta为:', F_beta)
    # 15、Jaccard相似系数
    Jaccard = metrics.jaccard_score(y_test, y_pre, average='weighted')
    print('Jaccard相似系数为:', Jaccard)
    # 16、调整互信息 AMI 输入参数是labels_true, labels_pred
    AMI = metrics.adjusted_mutual_info_score(y_test, y_pre)
    print('调整互信息AMI为:', AMI)
    # 17、标准化互信息NMI
    NMI = metrics.normalized_mutual_info_score(y_test, y_pre)
    print('标准化互信息NMI为:', NMI)
    # 18、方差回归得分EVS
    EVS = metrics.explained_variance_score(y_test, y_pre)
    print('方差回归得分EVS为:', EVS)
    # 19、标签排名平均精度LRAP
    # LRAP = metrics.label_ranking_average_precision_score(y_test,clf.predict(X_test)) 不支持多重
    # 20、Fowlkes-Mallows指数（FMI）定义为精度和召回率之间的几何平均值
    FMI = metrics.fowlkes_mallows_score(y_test, y_pre)
    print('Fowlkes-Mallows指数FMI为:', FMI)
    Evaluation_index = [ACU, Precison_Score, Recall, F1,
                        RMS, hammingloss, matthews, R2, BAS, F_beta, Jaccard,
                        AMI, NMI, EVS, FMI]
    Evaluation_index = pd.DataFrame(Evaluation_index)
    Evaluation_index = Evaluation_index.T
    Evaluation_index.columns = [
        '平均准确率ACU',
        '精准率Precision',
        '召回率Recall',
        'F1得分',
        '方均根误差RMS',
        '汉明损失HLoss',
        'Matthews相关系数',
        'R2系数',
        '平衡精度BAS',
        '加权谐波平均值F_beta',
        '相似系数Jaccard',
        '调整互信息AMI',
        '标准化互信息NMI',
        '方差回归得分EVS',
        'Fowlkes-Mallows指数FMI']
    return Evaluation_index


Data = pd.read_csv(
    r"C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/FirstLevelPCA.csv",
    encoding='gbk')
X_dataframe = Data.iloc[:, 0:-1]  # 分出数据和标签 此时是DataFrame格式
y_dataframe = Data.iloc[:, -1]
X = X_dataframe.values  # ndarray格式 样本数×维数
y_category = y_dataframe.values  # ndarray格式
Label = LabelEncoder()  # 初始化1个独热编码类
y = Label.fit_transform(y_category)  # 自动生成标签
# %%
'''train_test_split : 留出法'''
EIK = pd.DataFrame()
starttime = time()
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3)  # 训练数据比例7成
    clf = svm.SVC(kernel='linear', C=1, probability=True)
    clf.fit(X_train, y_train)  # 训练模型
    y_pre = clf.predict(X_test)
    Evaluation_index = EvaluationIndex(
        X_test, y_test, y_pre, clf, confusion_matrix=False)
    EIK = pd.concat([EIK, Evaluation_index], axis=0, ignore_index=True)
endtime = time()
Time = (endtime - starttime) / 100
# bw_method 太小过拟合,太大欠拟合 ,ind指定x轴位置转折 ind=[0.81,0.85,0.91,0.95,0.97,0.98]
ACU = EIK['平均准确率ACU']
fig, ax = plt.subplots()
ax.grid(axis='y')
ACU.plot.kde(ax=ax, linewidth=2.5, bw_method=0.3)
ACU.plot.hist(density=True, bins=12, color='c', ax=ax)
ax.set_xlabel('准确率')
ax.set_ylabel('频数')
ax.axis('tight')
labels = ["核密度估计图", "直方图"]
ax.legend(labels, loc='upper left')
ax1 = ax.twinx()
ax1.plot(ACU, ACU, 'b-v', linewidth=0.5)
meanACU = np.mean(ACU)
ax1.plot([min(ACU), 1.0], [meanACU, meanACU], 'r--')
ax1.text((min(ACU) + 1.0) / 2, meanACU + 0.005, "平均值 : " + str(round(meanACU, 5)),
         fontsize=14, verticalalignment='center', horizontalalignment='center')
ax1.text((min(ACU) + 1.0) / 2, min(ACU) + 0.006, "训练时间 : " + str(round(Time, 5)) +
         ' s', verticalalignment='center', horizontalalignment='center', fontsize=14)
ax1.axis('tight')
ax1.legend(('折线图', '平均值'), loc='center left')
ax1.fill_betweenx(
    np.arange(
        min(ACU) -
        0.05,
        1.02,
        0.05),
    1.0,
    1.02,
    alpha=0.2,
    color="r",
    lw=2)
ax1.set_xlim(min(ACU), 1.02)
ax1.set_ylim(min(ACU), 1.0)
ax1.text(
    (1.0 + 1.02) / 2,
    (min(ACU) + 1) / 2,
    "区域不存在",
    fontsize=10.5,
    family='SimHei',
    verticalalignment='center',
    horizontalalignment='center')
plt.title('线性核Linear留出法预测准确率分布图(训练次数:100次)')
fig.tight_layout()
plt.show()
# %%
'''K折交叉验证'''
state = np.random.get_state()  # 必须先打乱原来的数据集合标签集,否则初始的有序状态会导致交叉验证预测率很低
np.random.shuffle(X)
np.random.set_state(state)  # 保证样本和标签以相同的规律被打乱
np.random.shuffle(y)
n_spilits = 30
kf = KFold(n_splits=n_spilits)  # k折交叉验证 , 数据集划分为k等份
'''例如n_splits=3,那么把241份样本等分,各80份,分别记为C1、C2、C3,那么训练集轮流为[C2 C3],[C1 C3],[C1 C2],相应剩余的是测试集
交叉验证的前提是样本初始是无序的,如果是有序的,可能[C2 C3]中的C2对应标签012,C3对应567,C2则是34,训练集的标签012的模型对测试集的标签34是不能识别的'''
EIK = pd.DataFrame()
K = 0  # 用于确定当前训练的次数
for train_index, test_index in kf.split(X):
    train_X, train_y = X[train_index], y[train_index]
    test_X, test_y = X[test_index], y[test_index]
    idx = kf.get_n_splits()  # n_splits
    if idx:
        K = K + 1
    print(f'\n第{K}次的评价指标得分表为:')
    #print(f'第{K}次训练样本对应的索引为:', train_index, f'\n第{K}次测试样本对应的索引为:', test_index)
    clf1 = svm.SVC(kernel='linear', C=1, probability=True)
    clf1.fit(train_X, train_y)  # 训练模型
    y_pre1 = clf1.predict(test_X)
    Evaluation_indexkfold = EvaluationIndex(
        test_X, test_y, y_pre1, clf1, confusion_matrix=False)
    EIK = pd.concat([EIK, Evaluation_indexkfold], axis=0, ignore_index=True)
# %%
'''对于交叉验证而言还可以考虑不同正则化系数C的影响得到交叉验证得分估计图'''
'''模型评估 : 不同正则化系数交叉验证得分'''
C_s = np.logspace(-30, 30, 100)  # 10^(-10) ~ 10^0 , 分成十个,默认基准值10为底数
#C_s = np.linspace(0.000001,1,100,endpoint=False)
n_folds = 10
scores = list()  # 存放不同正则化系数的得分
scores_std = list()
clf = svm.SVC(kernel='linear', probability=True)
for C in C_s:
    clf.C = C
    this_scores = cross_val_score(
        clf, X, y, cv=n_folds, n_jobs=1)  # n_jobs用于进行计算的CPU数量
    scores.append(np.mean(this_scores))
    scores_std.append(np.std(this_scores))
# X轴采用对数刻度 , 解决图例一张图的方法
ax1, = plt.semilogx(C_s, scores)  # 绘制得分
ax2, = plt.semilogx(C_s, np.array(scores) +
                    np.array(scores_std), 'r--')  # 绘制得分的上包络线
ax3, = plt.semilogx(C_s, np.array(scores) -
                    np.array(scores_std), 'b--')  # 绘制得分的下包络线
plt.fill_between(
    C_s,
    np.array(scores) +
    np.array(scores_std),
    np.array(scores) -
    np.array(scores_std),
    alpha=0.1,
    color="r")  # 着色
plt.legend([ax1, ax2, ax3], ['交叉验证得分', '上包络线', '下包络线'], loc='best')
# 解决方案
ax = plt.gca()
locs, labels = plt.yticks()  # 刻度的位置和它使用的数值
# 每个位置上使用新的刻度值,可以将小数刻度变为指数刻度
plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
for tick in ax.xaxis.get_major_ticks():  # 获取图片坐标轴的主刻度,循环设置格式 解决指数坐标不能显示负号的问题
    tick.label1.set_fontproperties('stixgeneral')
# 以上代码复制粘贴即可
plt.ylabel('交叉验证得分')
plt.xlabel("正则化系数C")
plt.title(f"线性核Linear不同正则化系数的{n_folds}折交叉验证得分趋势图")
plt.axis('tight')
plt.show()
# %%
'''模型评估 : 交叉验证估计,可以不使用'''
K = 0
y_pred = cross_val_predict(clf, X, y)  # 直接得到交叉验证的预测值 , 可以不用
for i in range(len(y_pred)):
    if y_pred[i] == y[i]:
        K = K + 1
print('交叉验证估计的准确率为:', K / len(y_pred), '%')
# %%
'''模型评估 : 学习率曲线'''
'''反映的是不同训练样本个数时的得分,将这些样本的个数与得分建立关系即学习率曲线
X轴最大为给定的最大训练样本数,即(1-test_size)*241'''
'''用1个交叉验证生成器划分整体数据集K次,每一次划分都会有一个训练集和测试集,由n_spilit决定
第K次的训练集再不断拿出若干个数量不断增加的子集,子集占据训练集的比例由train_size决定
然后计算模型在对应的子训练集和测试集上的得分,最后在每种不同得子训练集下将K次训练集得得分和测试集得分分别平均
例如100个样本,第10次划分生成不同于之前9次的80个训练样本和20个测试样本,80个样本再按照0.1,0.3,0.5,0.7,0.9,1.0即8,24,40,56,72,80个训练样本训练模型去预测20个测试样本
每次预测都会有得分,这是第10次或者说第10列的得分 ; 然后所有交叉验证的分数按列求平均
至于某行,是80个样本都按某个比例如0.1,也就是这一行都是8个样本去预测20个测试集
训练子集的大小必须保证每个类至少1个样本,即最少8个样本,每个样本都是不同的标签'''
cv = ShuffleSplit(n_splits=10, test_size=0.1, random_state=0)
train_size = np.linspace(0.1, 1.0, 10)  # 控制训练集的子集比例
title = "线性核Linear不同训练集比例的10折交叉验证得分(固定正则化系数C=0.1)"
estimator = svm.SVC(kernel='linear', probability=True)
plt_learning_curve(
    estimator,
    title,
    X,
    y,
    ylim=(
        0.7,
        1.0),
    cv=cv,
    train_size=train_size)
# %%
'''模型评估 : 不同gamma的模型验证曲线'''
'''某个参数不断变化时每一个取值上计算出的模型再训练集和测试集上的得分'''
param_range = np.logspace(-30, 30, 100)
train_scores, test_scores = validation_curve(svm.SVC(
    kernel='linear'), X, y, param_name="gamma", param_range=param_range, cv=10, scoring="accuracy", n_jobs=1)
train_scores_mean = np.mean(train_scores, axis=1)  # 训练集得分
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)  # 测试集得分
test_scores_std = np.std(test_scores, axis=1)
# gamma是选择RBF函数作为kernel后，该函数自带的一个参数 gamma越大，支持向量越少，gamma值越小，支持向量越多。支持向量的个数影响训练与预测的速度
# gamma 与 高斯部分方差成反比,gamma越大方差越小,分布越集中,训练样本准确率很高但是泛化能力差 gamma=1/(2*方差)
plt.title(r"线性核Linear不同$\gamma$系数的交叉验证得分趋势图")
plt.xlabel(r"内核系数$\gamma$")
plt.ylabel("得分")
plt.ylim(0.0, 1.0)
lw = 1.5
# 训练集 主曲线
ax1, = plt.semilogx(param_range, train_scores_mean,
                    label="交叉验证-训练集得分", color="r", lw=lw)
ax2, = plt.semilogx(param_range, np.array(train_scores_mean) +
                    np.array(train_scores_std), 'b--', label='训练集上包络线', lw=0.5)  # 训练集上包络线
ax3, = plt.semilogx(param_range, np.array(train_scores_mean) -
                    np.array(train_scores_std), 'b--', label='训练集下包络线', lw=0.5)  # 训练集下包络线
# 训练集 上下包络线
plt.fill_between(
    param_range,
    train_scores_mean -
    train_scores_std,
    train_scores_mean +
    train_scores_std,
    alpha=0.2,
    color="r",
    lw=lw)
# 测试集 主曲线
ax4, = plt.semilogx(param_range, test_scores_mean,
                    label="交叉验证-测试集得分", color="g", lw=lw)
ax5, = plt.semilogx(param_range, np.array(test_scores_mean) +
                    np.array(test_scores_std), 'y--', label='测试集上包络线', lw=0.5)  # 测试集上包络线
ax6, = plt.semilogx(param_range, np.array(test_scores_mean) -
                    np.array(test_scores_std), 'y--', label='测试集下包络线', lw=0.5)  # 测试集下包络线
# 测试集 上下包络线
plt.fill_between(
    param_range,
    test_scores_mean -
    test_scores_std,
    test_scores_mean +
    test_scores_std,
    alpha=0.2,
    color="g",
    lw=lw)
minscore = min(
    np.min(
        np.array(train_scores_mean) -
        np.array(train_scores_std)),
    np.min(
        np.array(test_scores_mean) -
        np.array(test_scores_std)))
plt.ylim((minscore, 1))
plt.legend(loc='best')
# 不显示指数负号解决方案
ax = plt.gca()
locs, labels = plt.yticks()  # 刻度的位置和它使用的数值
# 每个位置上使用新的刻度值,可以将小数刻度变为指数刻度
plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
for tick in ax.xaxis.get_major_ticks():  # 获取图片坐标轴的主刻度,循环设置格式 解决指数坐标不能显示负号的问题
    tick.label1.set_fontproperties('stixgeneral')
plt.show()
# %%
'''模型评估 : 通过排序评估交叉验证的得分重要性 不常用'''
clf = svm.SVC(kernel='linear', C=1, probability=True)
# n_permutations 置换'y'的次数
# 没有置换目标的真实分数 、每个排列获得的分数 、近似于偶然获得分数的可能性(C + 1) / (n_permutations + 1)
# 其中C是其分数> =真实分数的排列数量 可能的最佳p值是1 /（n_permutations +1），最差的是1.0
score, permutation_scores, pvalues = permutation_test_score(
    clf, X, y, scoring='accuracy', cv=10, n_jobs=1, n_permutations=100)
#print(("分类得分 %s (pvalue : %s)" %(score,pvalues)))
permutation_score = pd.Series(permutation_scores)
fig, ax = plt.subplots()
ax.grid(axis='y')
# bw_method 太小过拟合,太大欠拟合 ,ind指定x轴位置转折 ind=[0.81,0.85,0.91,0.95,0.97,0.98]
permutation_score.plot.kde(ax=ax, linewidth=2.5, bw_method=0.13)
permutation_score.plot.hist(density=True, bins=12, color='c', ax=ax)
plt.title('线性核Linear排列分数直方图和核密度估计图')
plt.xlabel('排列分数')
plt.ylabel('比例%')
labels = ["核密度估计图", "直方图"]
plt.legend(labels)
plt.axis('tight')
plt.show()
# %%
'''留一法'''
'''留一法是k折交叉验证当中，k=n（n为数据集个数）的情形,相当于每次都只有1个测试样本,准确率=预测正确的个数/241'''
'''留一法由于每次只有一个测试样本,所以上述的评价参数表不能使用,而应比对每次留一时预测标签和实际标签是否相等,再用相等的次数除总的才是准确率'''
loo = LeaveOneOut()
K = 0
num = 0  # 计算预测和实际相等的次数
ACU = []  # 用于存放每次预测时  当前已经预测正确的个数与当前训练次数的比值, 例如可能训练5次3次准确,训练10次中6次准确,直到最终241次有几个样本准确是真实准确率
# 由此可以得到训练过程中ACU的变化曲线
EIK = pd.DataFrame()
starttime = time()
for train_index, test_index in loo.split(X):
    train_X, train_y = X[train_index], y[train_index]
    test_X, test_y = X[test_index], y[test_index]
    idx = loo.get_n_splits(X)  # n_splits = 241
    if idx:
        K = K + 1
    clf1 = svm.SVC(kernel='linear', C=1, probability=True)
    clf1.fit(train_X, train_y)  # 训练模型
    y_pre1 = clf1.predict(test_X)
    if y_pre1 == test_y:
        num = num + 1
    acu = num / K  # 当前预测正确的次数 / 训练的次数
    ACU.append(acu)
endtime = time()
Time = endtime - starttime
ACU = pd.Series(ACU)
fig, ax = plt.subplots()
ax.grid(axis='y')
ACU.plot.kde(ax=ax, linewidth=2.5, bw_method=0.3)
ACU.plot.hist(density=True, bins=12, color='c', ax=ax)
ax.set_xlabel('准确率')
ax.set_ylabel('频数')
ax.axis('tight')
labels = ["核密度估计图", "直方图"]
ax.legend(labels, loc='upper left')
ax1 = ax.twinx()
ACU = ACU.values
ax1.plot(ACU, ACU, 'b-v', linewidth=0.5)
meanACU = np.mean(ACU)
ax1.plot(ACU, label='准确率')
ax1.plot([0, len(ACU)], [meanACU, meanACU], 'r--', label='平均准确率')
ax1.plot([0, len(ACU)], [ACU[-1], ACU[-1]], 'g--', label='真实准确率')
ax1.fill_betweenx(
    np.arange(
        min(ACU) -
        0.05,
        1.02,
        0.05),
    1.0,
    1.02,
    alpha=0.2,
    color="r",
    lw=2)
ax1.text((min(ACU) + 1.0) / 2, meanACU + 0.002, "平均准确率 : " + str(round(meanACU, 5)),
         verticalalignment='center', horizontalalignment='center', fontsize=14)
ax1.text((min(ACU) + 1.0) / 2,
         ACU[-1] + 0.002,
         "真实准确率 : " + str(round(ACU[-1],
                                5)),
         verticalalignment='center',
         horizontalalignment='center',
         fontsize=14)
ax1.text(
    (1.0 + 1.02) / 2,
    (min(ACU) + 1) / 2,
    "区域不存在",
    fontsize=10.5,
    family='SimHei',
    verticalalignment='center',
    horizontalalignment='center')
ax1.text((min(ACU) + 1.0) / 2, min(ACU) + 0.002, "训练时间 : " + str(round(Time, 5)) +
         ' s', verticalalignment='center', horizontalalignment='center', fontsize=14)
ax1.axis('tight')
ax1.legend(('折线图', '平均值'), loc='center left')
ax1.set_xlim(min(ACU), 1.02)
ax1.set_ylim(min(ACU), 1.0)
fig.tight_layout()
plt.title(f'线性核Linear留1法准确率变化曲线(训练次数:{len(ACU)}次)')
plt.show()
# %%
'''留P法对应留一法,区别在于每次留下P个测试集'''
p = 2
lpo = LeavePOut(p=p)
K = 0
EIK = pd.DataFrame()
SumACU = 0
ACU = []
starttime = time()
for train_index, test_index in lpo.split(X):
    train_X, train_y = X[train_index], y[train_index]
    test_X, test_y = X[test_index], y[test_index]
    idx = lpo.get_n_splits(X)  # n_splits = 241 - p
    if idx:
        K = K + 1  # 28920
    clf1 = svm.SVC(kernel='linear', C=1, probability=True)
    clf1.fit(train_X, train_y)  # 训练模型
    y_pre1 = clf1.predict(test_X)
    pp = 0
    for j in range(len(y_pre1)):
        if y_pre1[j] == test_y[j]:
            pp = pp + 1
    acu = pp / len(y_pre1)  # p = 2 只有0,0.5和1的取值
    # 例如留3个,测试集3个都对算一次预测正确,2个正确算2/3次正确,1个正确算1/3次正确,或者不正确为0次
    SumACU = SumACU + acu  # 计算到当前训练次数的累计正确次数
    # 当前的准确率
    # 例如当前训练次数10次,可能全部正确的2个,2/3正确的3个,1/3正确的1个,其它不正确,那么总的准确率为[2×1+3×(2/3)+1×(1/3)]/10=13/30
    acu = SumACU / K
    ACU.append(acu)
endtime = time()
Time = endtime - starttime
ACU = pd.Series(ACU)
fig, ax = plt.subplots()
ax.grid(axis='y')
ACU.plot.kde(ax=ax, linewidth=2.5, bw_method=0.3)
ACU.plot.hist(density=True, bins=12, color='c', ax=ax)
ax.set_xlabel('准确率')
ax.set_ylabel('频数')
ax.axis('tight')
labels = ["核密度估计图", "直方图"]
ax.legend(labels, loc='upper left')
ax1 = ax.twinx()
ACU = ACU.values
ax1.plot(ACU, ACU, 'b-v', linewidth=0.5)
meanACU = np.mean(ACU)
ax1.plot(ACU, label='准确率')
ax1.plot([0, len(ACU)], [meanACU, meanACU], 'r--', label='平均准确率')
ax1.plot([0, len(ACU)], [ACU[-1], ACU[-1]], 'g--', label='真实准确率')
ax1.fill_betweenx(
    np.arange(
        min(ACU),
        1.02,
        0.005),
    1.0,
    1.02,
    alpha=0.2,
    color="r",
    lw=2)
ax1.text((min(ACU) + 1.0) / 2, meanACU + 0.002, "平均准确率 : " + str(round(meanACU, 5)),
         verticalalignment='center', horizontalalignment='center', fontsize=14)
ax1.text((min(ACU) + 1.0) / 2,
         ACU[-1] + 0.002,
         "真实准确率 : " + str(round(ACU[-1],
                                5)),
         verticalalignment='center',
         horizontalalignment='center',
         fontsize=14)
ax1.text(
    (1.0 + 1.02) / 2,
    (min(ACU) + 1) / 2,
    "区域不存在",
    fontsize=10.5,
    family='SimHei',
    verticalalignment='center',
    horizontalalignment='center')
ax1.text((min(ACU) + 1.0) / 2, min(ACU) + 0.002, "训练时间 : " + str(round(Time, 5)) +
         ' s', verticalalignment='center', horizontalalignment='center', fontsize=14)
ax1.axis('tight')
ax1.legend(('折线图', '平均值'), loc='center left')
ax1.set_xlim(min(ACU), 1.02)
ax1.set_ylim(min(ACU), 1.0)
fig.tight_layout()
plt.title(f'线性核Linear留2法准确率变化曲线(训练次数:{len(ACU)}次)')
plt.show()
# %%
'''P次K折交叉验证'''
'''上述的交叉验证本身也只是做了一次,即一次交叉验证对K份不同的训练-测试集进行训练,原理上相当于对留出法进行了K次试验
但是K折交叉验证本身也需要多次,即P次,最终可以认为是留出法做了P×K次的试验
最典型的是: 10次10折交叉验证,RepeatedKFold方法可以控制交叉验证的次数'''
state = np.random.get_state()  # 必须先打乱原来的数据集合标签集,否则初始的有序状态会导致交叉验证预测率很低
np.random.shuffle(X)
np.random.set_state(state)  # 保证样本和标签以相同的规律被打乱
np.random.shuffle(y)
# random_state=0可以保证每次K折交叉验证的随机打乱规律不同
kf = RepeatedKFold(n_splits=10, n_repeats=10, random_state=0)
EIK = pd.DataFrame()
P = 0
for train_index, test_index in kf.split(X):
    idx = kf.get_n_splits()
    if idx:
        P = P + 1
    print(f'\n第{P}次训练的评价指标分数表为:')
    train_X, train_y = X[train_index], y[train_index]
    test_X, test_y = X[test_index], y[test_index]
    clf2 = svm.SVC(kernel='linear', C=1, probability=True)
    clf2.fit(train_X, train_y)  # 训练模型
    y_pre2 = clf2.predict(test_X)
    Evaluation_indexkfold = EvaluationIndex(
        test_X,
        test_y,
        y_pre2,
        clf2,
        confusion_matrix=False,
        title='线性核Linear')
    EIK = pd.concat([EIK, Evaluation_indexkfold], axis=0, ignore_index=True)
meanACU = EIK.mean(axis=0)
# %%
'''管道机制实现了对全部步骤的流式化封装和管理
管道机制更像是编程技巧的创新，而非算法的创新'''
'''使用RBFSampler和Nystroem来近似RBF内核的特征图
比较了使用原始空间中的线性SVM，使用近似映射的线性SVM和使用内核化SVM的结果
显示了不同数量的蒙特卡洛采样的时间和精度（在RBFSampler的情况下，它使用随机傅里叶特征）
和训练集的不同大小的子集（对于Nystroem）用于近似映射
请注意，这里的数据集不足以显示核近似的好处，因为精确的SVM仍然相当快
对更多维度进行采样显然会带来更好的分类结果，但代价更高。 这意味着在运行时间和精度之间需要权衡，这由参数n_components给出
①RBFSampler和Nystroem替代RBF核函数
②同时还对比了线性核
③训练集的不同大小的子集'''
X = pd.read_csv(
    r"C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/Circuit_Breaker_FirstLevelLabel.csv",
    encoding='gbk')
X = X.drop(['Category'], axis=1)  # 删除标签列
X = X.values  # 划分训练和测试的参数是数组格式
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
n_spilit = len(X_train)
n_spilit_size = np.linspace(2, n_spilit, n_spilit - 1)  # 2,3,4,...192 至少2份
sample_sizes = np.arange(1, n_spilit)
kernel_svm = svm.SVC(gamma=.2)  # 不指定默认是rbf
linear_svm = svm.LinearSVC()  # 与指定'linear'是一样的
Linear_time = []
Kernel_time = []
Linear_score = []
Kernel_score = []
starttime = time()
for i in range(len(n_spilit_size)):  # 依次使用训练集的2,3,4,...192个样本
    # 以下四行程序是为了保证每次使用训练集其中的i个样本也是随机的
    state = np.random.get_state()  # 必须先打乱原来的数据集合标签集,否则初始的有序状态会导致交叉验证预测率很低
    np.random.shuffle(X_train)
    np.random.set_state(state)  # 保证样本和标签以相同的规律被打乱
    np.random.shuffle(y_train)
    # print(int(n_spilit_size[i]))
    X_train_son = X_train[0:int(n_spilit_size[i]), :]  # 前 i 行 作为训练集 的子集
    y_train_son = y_train[0:int(n_spilit_size[i])]  # 前 i 行 作为训练集 的子集
    Kernel_svm_time = time()  # rbf核函数的程序开始时间
    kernel_svm.fit(X_train_son, y_train_son)  # 训练集拟合
    kernel_svm_score = kernel_svm.score(X_test, y_test)  # 测试集得分
    kernel_svm_time = time() - Kernel_svm_time  # rbf核函数运行消耗的时间
    Linear_svm_time = time()  # 线性核函数的程序开始时间
    linear_svm.fit(X_train, y_train)  # 训练集拟合
    linear_svm_score = linear_svm.score(X_test, y_test)  # 测试集得分
    linear_svm_time = time() - Linear_svm_time  # 线性核函数运行消耗的时间
    Linear_score.append(linear_svm_score)  # 存放不同训练子集数量预测测试集的时间和分数
    Linear_time.append(linear_svm_time)
    Kernel_score.append(kernel_svm_score)
    Kernel_time.append(kernel_svm_time)
'''线性核 : 从内核近似创建管道'''
feature_map_fourier = RBFSampler(gamma=.2, random_state=1)  # 傅里叶变换的蒙特卡洛近似
feature_map_nystroem = Nystroem(
    gamma=.2, random_state=1)  # 使用数据的子集为任意内核构造一个近似特征图
fourier_approx_svm_rbf = pipeline.Pipeline(
    [("feature_map", feature_map_fourier), ("svm", svm.SVC(gamma=.2))])  # 基于RBF构建第一个管道
nystroem_approx_svm_rbf = pipeline.Pipeline(
    [("feature_map", feature_map_nystroem), ("svm", svm.SVC(gamma=.2))])  # 基于RBF构建第二个管道
fourier_approx_svm_linear = pipeline.Pipeline(
    [("feature_map", feature_map_fourier), ("svm", svm.LinearSVC())])  # 基于Linear构建第一个管道
nystroem_approx_svm_linear = pipeline.Pipeline(
    [("feature_map", feature_map_nystroem), ("svm", svm.LinearSVC())])  # 基于Linear构建第二个管道
#sample_sizes = 30 * np.arange(1, 10)
fourier_scores_rbf = []  # 分别存放fourier和nystroem内核估计的分数和消耗时间
nystroem_scores_rbf = []
fourier_times_rbf = []
nystroem_times_rbf = []
fourier_scores_linear = []  # 分别存放fourier和nystroem内核估计的分数和消耗时间
nystroem_scores_linear = []
fourier_times_linear = []
nystroem_times_linear = []
for D in sample_sizes:
    # 基于RBF
    fourier_approx_svm_rbf.set_params(
        feature_map__n_components=D)  # 管道1设置参数 ,D不能超过训练集的数量
    nystroem_approx_svm_rbf.set_params(feature_map__n_components=D)
    start = time()
    nystroem_approx_svm_rbf.fit(X_train, y_train)  # rbf的nystroem时间
    nystroem_times_rbf.append(time() - start)
    start = time()
    fourier_approx_svm_rbf.fit(X_train, y_train)  # rbf的fourier时间
    fourier_times_rbf.append(time() - start)
    fourier_score = fourier_approx_svm_rbf.score(
        X_test, y_test)  # rbf的nystroem得分
    nystroem_score = nystroem_approx_svm_rbf.score(
        X_test, y_test)  # rbf的fourier得分
    nystroem_scores_rbf.append(nystroem_score)
    fourier_scores_rbf.append(fourier_score)
    # 基于Linear
    fourier_approx_svm_linear.set_params(
        feature_map__n_components=D)  # 管道1设置参数 ,D不能超过训练集的数量
    nystroem_approx_svm_linear.set_params(feature_map__n_components=D)
    start = time()
    nystroem_approx_svm_linear.fit(X_train, y_train)  # rbf的nystroem时间
    nystroem_times_linear.append(time() - start)
    start = time()
    fourier_approx_svm_linear.fit(X_train, y_train)  # rbf的fourier时间
    fourier_times_linear.append(time() - start)
    fourier_score = fourier_approx_svm_linear.score(
        X_test, y_test)  # rbf的nystroem得分
    nystroem_score = nystroem_approx_svm_linear.score(
        X_test, y_test)  # rbf的fourier得分
    nystroem_scores_linear.append(nystroem_score)
    fourier_scores_linear.append(fourier_score)

plt.figure(figsize=(12, 8.4))
accuracy = plt.subplot(211)  # 准确率
timescale = plt.subplot(212)  # 耗时
# 基于不同核函数的两种估计的时间和准确率
accuracy.plot(sample_sizes, nystroem_scores_rbf, label="基于RBF核的奈斯特罗姆近似--准确率")
timescale.plot(
    sample_sizes,
    nystroem_times_rbf,
    '--',
    label='基于RBF核的奈斯特罗姆近似-模型训练时间')
accuracy.plot(sample_sizes, fourier_scores_rbf, label="基于RBF核的蒙特卡洛近似-准确率")
timescale.plot(
    sample_sizes,
    fourier_times_rbf,
    '--',
    label='基于RBF核的蒙特卡洛近似-模型训练时间')

accuracy.plot(
    sample_sizes,
    nystroem_scores_linear,
    label="基于Linear核的奈斯特罗姆近似-准确率")
timescale.plot(sample_sizes, nystroem_times_linear,
               '--', label='基于Linear核的奈斯特罗姆近似-模型训练时间')
accuracy.plot(
    sample_sizes,
    fourier_scores_linear,
    label="基于Linear核的蒙特卡洛近似-准确率")
timescale.plot(
    sample_sizes,
    fourier_times_linear,
    '--',
    label='基于Linear核的蒙特卡洛近似-模型训练时间')
# 没有近似的线性核和RBF核的时间和准确率
accuracy.plot(sample_sizes, Linear_score, label="普通线性核-准确率")
timescale.plot(sample_sizes, Linear_time, '--', label='普通线性核-模型训练时间')
accuracy.plot(sample_sizes, Kernel_score, label="普通RBF核-准确率")
timescale.plot(sample_sizes, Kernel_time, '--', label='普通RBF核-模型训练时间')
# 精确的rbf和线性内核的水平线
#accuracy.plot([sample_sizes[0], sample_sizes[-1]],[linear_svm_score, linear_svm_score], label="线性核平均准确率")
#timescale.plot([sample_sizes[0], sample_sizes[-1]],[linear_svm_time, linear_svm_time], '--', label='线性核平均模型训练时间')
#accuracy.plot([sample_sizes[0], sample_sizes[-1]],[kernel_svm_score, kernel_svm_score], label="RBF核平均准确率")
#timescale.plot([sample_sizes[0], sample_sizes[-1]],[kernel_svm_time, kernel_svm_time], '--', label='RBF核平均模型训练时间')
# 数据集维度的垂直线 = 23 这个可以观察nystroem_scores得分在第23个明显上升
# accuracy.plot([23, 23], [min(nystroem_scores[0],fourier_scores[0]), 1], label="分界线") # 这里观察最低得分0.4794 ,即垂直线的两个点,横坐标都是4,纵坐标分别是0.47和1
# 设置图片参数
minscore = min(
    np.min(fourier_scores_rbf),
    np.min(nystroem_scores_rbf),
    np.min(Kernel_score),
    np.min(Linear_score),
    np.min(fourier_scores_linear),
    np.min(nystroem_scores_linear))  # 找到6种评价分数各自最小的最小那个作为y轴最小值
maxscore = max(np.max(fourier_scores_rbf), np.max(nystroem_scores_rbf),
               np.max(Kernel_score), np.max(Linear_score),
               np.max(fourier_scores_linear), np.max(nystroem_scores_linear))
accuracy.set_ylim(minscore, maxscore)
accuracy.set_xlim(0, len(sample_sizes))
accuracy.set_ylabel("准确率")
accuracy.set_xlabel("训练集样本数量/个")
accuracy.legend(loc='best')
accuracy.set_title("基于Linear和RBF核的奈斯特罗姆近似与蒙特卡洛近似的准确率变化图")
mintime = min(np.min(fourier_times_rbf), np.min(nystroem_times_rbf),
              np.min(Kernel_time), np.min(Linear_time),
              np.min(fourier_times_linear), np.min(nystroem_times_linear))
maxtime = max(np.max(fourier_times_rbf), np.max(nystroem_times_rbf),
              np.max(Kernel_time), np.max(Linear_time),
              np.max(fourier_times_linear), np.max(nystroem_times_linear))
timescale.set_ylim(mintime, maxtime)
timescale.set_xlim(0, len(sample_sizes))
timescale.set_xlabel("训练集样本数量/个")
timescale.set_ylabel("模型训练时间/s")
timescale.legend(loc='best')
timescale.set_title("基于Linear和RBF核的奈斯特罗姆近似与蒙特卡洛近似的模型训练时间变化图")
plt.show()
endtime = time()
print('程序花费时间为 : ', endtime - starttime)
# %%
# ROC-AUC
# 输入值真实标签及其得分 y_true, y_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = svm.SVC(kernel='linear', C=1, probability=True)
clf.fit(X_train, y_train)
proba = clf.predict_proba(X_test)  # 测试样本被预测为各种标签对应的概率矩阵
'''生成类似于二进制的新标签矩阵,将概率矩阵每行最大的概率改为1,其余为0'''
#Rowmax = []
# for i in range(len(proba)) :
#    row = list( proba [i]  )
#    rowmax  = row.index(max(row)) # 找到该行最大值对应的索引
#    Rowmax.append(rowmax)
Y = proba.copy()  # 防止改变源列表
# 不应找到最大值,而是阈值 ,即人为大于0.5的都是1
Threvalues = 0.5
for i in range(len(Y)):
    for j in range(Y.shape[1]):
        if proba[i, j] > Threvalues:  # 如果第i行的第j列到达了该行最大值的那一列,将该元素置为1,否则为0
            Y[i, j] = 1
        else:
            Y[i, j] = 0
# 方法1
Fpr = []
Tpr = []
AUC = []
for i in range(Y.shape[1]):  # 对proba和Y的每一列绘制roc曲线
    fpr, tpr, thresolds = metrics.roc_curve(Y[:, i], proba[:, i])
    auc = metrics.auc(fpr, tpr)
    plt.plot(
        fpr,
        tpr,
        color='darkorange',
        lw=2,
        label=f'第{i}类的ROC曲线(曲线面积 = %0.2f)' %
        auc)
    plt.legend()
    Fpr.append(fpr)
    Tpr.append(tpr)
    AUC.append(auc)
Fpr = pd.DataFrame(Fpr)
Tpr = pd.DataFrame(Tpr)
meanAUC = np.mean(AUC)
meantpr = Tpr.mean(axis=0)
meanfpr = Fpr.mean(axis=0)
plt.plot(
    meanfpr,
    meantpr,
    label='平均ROC曲线(曲线面积 = %0.2f)' %
    meanAUC,
    color='r',
    linestyle='--')  # 方法1
plt.legend(loc='upper right')
plt.title('所有类别的ROC曲线及其面积')
plt.show()
# 方法2 把Y和proba按行展开 ,不计算平均直接得到roc曲线
RowY = pd.DataFrame()
RowProba = pd.DataFrame()
for i in range(len(Y)):
    rowY = pd.DataFrame(Y[i, :]).T
    rowProba = pd.DataFrame(proba[i, :]).T
    RowY = pd.concat([RowY, rowY], axis=1, ignore_index=True)
    RowProba = pd.concat([RowProba, rowProba], axis=1, ignore_index=True)
for i in range(RowY.shape[1]):
    RowY.iloc[:, i] = np.int(RowY.iloc[:, i])
fpr, tpr, thresolds = metrics.roc_curve(
    RowY.T.values, RowProba.T.values)  # 必须是列向量
auc = metrics.auc(fpr, tpr)
plt.plot(fpr, tpr, label='ROC曲线(曲线面积 = %0.2f)' %
         auc, color='r', linestyle='-')  # 方法1
plt.legend(loc='upper right')
plt.title('ROC曲线及其面积')
plt.show()
# %%
