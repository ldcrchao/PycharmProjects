#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : PCA_CircuitBreaker.py
@Author : chenbei
@Date : 2020/12/17 10:33
'''
from matplotlib.pylab import mpl
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman'] # 设置字体风格,必须在前然后设置显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文的命令
mpl.rcParams['axes.unicode_minus'] = False  #  显示负号的命令
from sklearn.decomposition import PCA
#from sklearn.cross_decomposition import CCA # CCA用于执行监督降维
#from sklearn.datasets import make_multilabel_classification # 生成随机的多标签分类问题
#from sklearn.multiclass import OneVsRestClassifier # 一对多分类器 , 每个类分配一个二分类器
#from sklearn.preprocessing import LabelEncoder
#from sklearn.svm import SVC
import seaborn as sns
import pandas as pd
import numpy as np
#%% 一级标签 8 分类问题
X = pd.read_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/Circuit_Breaker_FirstLevelLabel.csv",encoding='gbk')#解决不能识别希腊字母的问题
model=PCA(n_components=2)
X_Features = X.iloc[:,0:-1] # 没有分类标签的纯数据
X_new=model.fit_transform(X_Features) # 训练并转换
X_new_dataframe = pd.DataFrame(X_new)
Category = X.iloc[:,-1] # 取出标签
X_new_category = pd.concat([X_new_dataframe,Category],axis=1,ignore_index= True) # 按列拼接
X_new_category.columns = ['PCA1','PCA2','Category'] # 重命名
#X_new_category.to_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/FirstLevelPCA.csv",encoding='gbk',index=False)
noise_variance = model.noise_variance_
score = model.score(X_Features)
singular_value = model.singular_values_
print('噪声协方差为:',noise_variance)
print('似然平均值为:',score)
print('奇异值为:',singular_value)
plt.figure(figsize=(7.8,3.8),dpi=600) # 设置图片大小、精度
fig1 = sns.scatterplot(data=X_new_category,x='PCA1',y='PCA2',hue='Category') # 主成分降维后的特征量的散点图可视化
fig1.set_xlabel('PCA1',fontsize=10.5) # 设置字体大小
fig1.set_ylabel('PCA2',fontsize=10.5)
plt.title('主成分散点图(n_conponents=2)')
plt.text(0,1.0 ,"噪声协方差 : "+str(round(noise_variance,5)) ,fontsize=10.5,verticalalignment='center',horizontalalignment='center',family='SimHei')
plt.text(0,1.5 ,"似然平均值 : "+str(round(score,5)) ,fontsize=10.5,verticalalignment='center',horizontalalignment='center',family='SimHei')
plt.tick_params(axis='both',which='major',labelsize=10.5) # 设置刻度
plt.tight_layout()
plt.show()

'''取出4个也是一样的，为了得到主成分方差贡献率，专门一段代码用于绘制方差贡献率图'''
model1 = PCA(n_components=4)
X_new1 = model1.fit_transform(X_Features)
X_new1_dataframe = pd.DataFrame(X_new1)

np.set_printoptions(precision=5)
ratio1 = np.around(model1.explained_variance_ratio_,5)# 主成分降维的方差贡献率
component_nums = model1.n_components_
plt.figure(figsize=(7.8,3.8),dpi=600)
reats = plt.bar(range(component_nums),ratio1) # 所有柱子
plt.ylabel('百分比')
plt.title('主成分方差贡献率')
plt.xlabel('维度')
plt.xticks(ticks=[0,1,2,3],labels=['PCA1','PCA2','PCA3','PCA4'])
plt.tick_params(axis='both',which='major',labelsize=10.5)
for reat in reats : # 每一个柱子循环标注数值大小
    reat_height = reat.get_height() # 获取柱子高度
    plt.text(reat.get_x() + reat.get_width()/2,reat_height,str(reat_height),size=10.5,ha='center',va='bottom')
plt.tight_layout()
plt.show()
# 保存数据
#X_new_category.to_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/PCA.csv",encoding='gbk',index=False)
#%% 二级标签 5 分类问题
X = pd.read_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/Circuit_Breaker_SecondLevelLabel.csv",encoding='gbk')#解决不能识别希腊字母的问题
model=PCA(n_components=2)
X_Features = X.iloc[:,0:-1] # 没有分类标签的纯数据
X_new=model.fit_transform(X_Features) # 训练并转换
X_new_dataframe = pd.DataFrame(X_new)
Category = X.iloc[:,-1] # 取出标签
X_new_category = pd.concat([X_new_dataframe,Category],axis=1,ignore_index= True) # 按列拼接
X_new_category.columns = ['PCA1','PCA2','Category'] # 重命名
#X_new_category.to_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/SecondLevelPCA.csv",encoding='gbk',index=False)
noise_variance = model.noise_variance_
score = model.score(X_Features)
singular_value = model.singular_values_
print('噪声协方差为:',noise_variance)
print('似然平均值为:',score)
print('奇异值为:',singular_value)
plt.figure(figsize=(7.8,3.8),dpi=600) # 设置图片大小、精度
fig1 = sns.scatterplot(data=X_new_category,x='PCA1',y='PCA2',hue='Category') # 主成分降维后的特征量的散点图可视化
fig1.set_xlabel('PCA1',fontsize=10.5) # 设置字体大小
fig1.set_ylabel('PCA2',fontsize=10.5)
plt.text(0,1.0 ,"噪声协方差 : "+str(round(noise_variance,5)) ,fontsize=10.5,verticalalignment='center',horizontalalignment='center',family='SimHei')
plt.text(0,1.5 ,"似然平均值 : "+str(round(score,5)) ,fontsize=10.5,verticalalignment='center',horizontalalignment='center',family='SimHei')
plt.title('主成分散点图(n_conponents=2)')
plt.tick_params(axis='both',which='major',labelsize=10.5) # 设置刻度
plt.tight_layout()
plt.show()

'''取出4个也是一样的，为了得到主成分方差贡献率，专门一段代码用于绘制方差贡献率图'''
model1 = PCA(n_components=4)
X_new1 = model1.fit_transform(X_Features)
X_new1_dataframe = pd.DataFrame(X_new1)

np.set_printoptions(precision=5)
ratio1 = np.around(model1.explained_variance_ratio_,5)# 主成分降维的方差贡献率
component_nums = model1.n_components_
plt.figure(figsize=(7.8,3.8),dpi=600)
reats = plt.bar(range(component_nums),ratio1) # 所有柱子
plt.ylabel('百分比')
plt.title('主成分方差贡献率')
plt.xlabel('维度')
plt.xticks(ticks=[0,1,2,3],labels=['PCA1','PCA2','PCA3','PCA4'])
plt.tick_params(axis='both',which='major',labelsize=10.5)
for reat in reats : # 每一个柱子循环标注数值大小
    reat_height = reat.get_height() # 获取柱子高度
    plt.text(reat.get_x() + reat.get_width()/2,reat_height,str(reat_height),size=10.5,ha='center',va='bottom')
plt.tight_layout()
plt.show()


