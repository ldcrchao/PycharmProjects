#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : MLP.py
@Author : chenbei
@Date : 2020/12/22 19:35
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


