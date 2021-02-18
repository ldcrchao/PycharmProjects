#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : Pycarnet_CircuitBreaker.py
@Author : chenbei
@Date : 2020/12/22 15:22
'''
from pycaret.classification import *
import pandas as pd
#%%
# 没有主成分降维 的 1级标签 8分类问题
Data = pd.read_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/Circuit_Breaker_FirstLevelLabel.csv",encoding='gbk')
clf = setup(data = Data , target = 'Category',train_size=0.7,n_jobs=1,
            numeric_imputation = 'mean',categorical_imputation = 'constant',feature_selection_threshold= 0.8)
best = compare_models()
#%%
# 没有主成分降维 的 2级标签 5分类问题
Data = pd.read_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/Circuit_Breaker_SecondLevelLabel.csv",encoding='gbk')
clf = setup(data = Data , target = 'Category',train_size=0.7,n_jobs=1,
            numeric_imputation = 'mean',categorical_imputation = 'constant',feature_selection_threshold= 0.8)
best = compare_models()
#%%
# 主成分降维 的 1级标签 8分类问题
Data = pd.read_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/Circuit_Breaker_FirstLevelLabel.csv",encoding='gbk')
clf = setup(data = Data , target = 'Category',train_size=0.7,n_jobs=1,
            numeric_imputation = 'mean',categorical_imputation = 'constant',feature_selection_threshold= 0.8,
            pca=True ,pca_method='kernel',pca_components=2 )
best = compare_models()
#%%
# 主成分降维 的 2级标签 5分类问题
Data = pd.read_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/Circuit_Breaker_SecondLevelLabel.csv",encoding='gbk')
clf = setup(data = Data , target = 'Category',train_size=0.7,n_jobs=1,
            numeric_imputation = 'mean',categorical_imputation = 'constant',feature_selection_threshold= 0.8,
            pca=True ,pca_method='kernel',pca_components=2)
best = compare_models()
#%%
# 主成分降维 的 1级标签 8分类问题 但是用的自行的pca数据
Data = pd.read_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/FirstLevelPCA.csv",encoding='gbk')
clf = setup(data = Data , target = 'Category',train_size=0.7,sampling= True,n_jobs=1,
            numeric_imputation = 'mean',categorical_imputation = 'constant',feature_selection_threshold= 0.8,)
best = compare_models()
#%%
# 主成分降维 的 2级标签 5分类问题 但是用的自行的pca数据
Data = pd.read_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/SecondLevelPCA.csv",encoding='gbk')
clf = setup(data = Data , target = 'Category',train_size=0.7,sampling= True,n_jobs=1,
            numeric_imputation = 'mean',categorical_imputation = 'constant',feature_selection_threshold= 0.8)
best = compare_models()