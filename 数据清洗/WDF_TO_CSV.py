#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : WDF_TO_CSV.py
@Author : chenbei
@Date : 2020/10/20 18:30
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
from Car import Troubleshooting
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文的命令
mpl.rcParams['axes.unicode_minus'] = False  #  显示负号的命令
def contact_csv(filename):
    # filename = "csv_线圈不正常"
    address = "C:/Users/chenbei/Desktop/钢/数据/"
    address = address + filename
    temp = ([ str(x)  for x in range(101)])
    Address = [[] for _ in range(101)]
    #Data = pd.DataFrame(columns=[str(x) for x in range(101)])
    Data = pd.DataFrame()
    for i in range(101):
        Address[i] = address +"/"+ temp[i] +".csv"
        data = pd.read_csv(Address[i])
        data.columns = ['0','data']
        del data['0']
        data.columns = [str(i)]
        Data = pd.concat([Data,data],axis=1,ignore_index= True)
    return  Data
def nan_to_zero(DataFrame):
    for i in range(DataFrame.shape[1]):
        for j in range(DataFrame.shape[0]):
            if pd.isnull(DataFrame.iloc[j, i]):
                DataFrame.iloc[j, i] = 0
    return DataFrame
def time_node():
    now = datetime.datetime.now()
    return now
def savecsv(DataFrame,filename):
    #filename = "csv_线圈正常/线圈卡涩1.csv"
    address = "C:/Users/chenbei/Desktop/钢/数据/"
    address = address + filename
    DataFrame.to_csv(address, index=False)
def Resample(DataFrame,samplerate):
    timenow = time_node()
    Index = pd.date_range(timenow, periods=len(DataFrame), freq='S')
    DataFrame.index = Index
    DataFrame = DataFrame.resample(samplerate).sum()
    for i in range(DataFrame.shape[1]):
        for j in range (DataFrame.shape[0]):
            DataFrame.iloc[j,i] = DataFrame.iloc[j,i] / max(DataFrame.iloc[:,i])
    return DataFrame
def selectusefuldata(DataFrame):
    DataFrame = DataFrame[200:800]
    return  DataFrame
def ColumName(name,length):
    # columnnames = ColumName("energy",32)
    namelist = [[]for _ in range(length)]
    temp = ([str(x) for x in range(length)])
    for i in range(length):
        namelist[i] = name + '_' + temp[i]
    return namelist
#%%
'''不正常信号'''
Data = contact_csv("csv_线圈不正常") # 未进行处理的有nan值的原始文件
Data = nan_to_zero(Data)
savecsv(Data,"imnormal.csv") # imnormal.csv 无nan值的原始文件
Data = pd.read_csv("C:/Users/chenbei/Desktop/钢/数据/csv_线圈不正常/线圈卡涩0.csv")
# 线圈卡涩0.csv 是已经手动处理好的没有nan值也没有大量的0值,的原始文件用于matlab使用
Data = Resample(Data,'10S') # 无nan值的重采样的文件
savecsv(Data,"线圈卡涩1.csv")
#%%
Data = pd.read_csv("C:/Users/chenbei/Desktop/钢/数据/csv_线圈不正常/imnormal.csv") # 读取原始文件,不使用线圈卡涩.csv和线圈卡涩1.csv
Feature  = Troubleshooting.feature_extract.WaveletAlternationEnhance(Data,'db8',5,plot=True,which=2)
savecsv(Feature,"csv_线圈不正常/energy_values_imnormal.csv") # 小波特征量文件
#%%
'''正常信号'''
Data = contact_csv("csv_线圈正常") # 未进行处理的有nan值的原始文件
Data = nan_to_zero(Data)
savecsv(Data,"csv_线圈正常/normal.csv") # normal.csv 无nan值的原始文件
#%%
Data = pd.read_csv("C:/Users/chenbei/Desktop/钢/数据/csv_线圈正常/normal.csv")
Feature  = Troubleshooting.feature_extract.WaveletAlternationEnhance(Data,'db8',5,plot=True,which=2)
savecsv(Feature,"csv_线圈正常/energy_values_normal.csv")




