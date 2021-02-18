#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : CSVdata_TO_CSVFeatures.py
@Author : chenbei
@Date : 2020/11/19 15:48
'''
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from Car import Troubleshooting
which ='220kase'
newaddress = "C:/Users/chenbei/Desktop/钢/线圈数据-20k/"+which+"/csv/data.csv" # 处理文件夹下所有30组的信号
kkadress ="C:/Users/chenbei/Desktop/钢/线圈数据-20k/极大值极小值/加了标签且所有样本都合并/"
#kadress = "C:/Users/chenbei/Desktop/钢/线圈数据-20k/180/csv/" #220/220chuan10需要调整最小值在>300而不是400，否则找不到最小值
def contact_csv(address):
    #address = "C:/Users/chenbei/Desktop/钢/线圈数据-20k/160/csv/"
    file = os.listdir(address)
    temp = ([ str(x)  for x in range(len(file))])
    Address = [[] for _ in range(len(file))]
    Data = pd.DataFrame()
    for i in range(len(file)):
        Address[i] = address  + temp[i] + ".csv"
        data = pd.read_csv(Address[i])
        data.columns = ['0','data']
        del data['0']
        data.columns = [str(i)]
        Data = pd.concat([Data,data],axis=1,ignore_index= True)
    return  Data
#%%
data = pd.read_csv(newaddress) # 1100×30
x_max = [[] for _ in range(data.shape[1])] # 30组的极大值横坐标
y_max = [[] for _ in range(data.shape[1])] # 30组的极大值纵坐标
x_min = [[] for _ in range(data.shape[1])] # 30组的极小值横坐标
y_min = [[] for _ in range(data.shape[1])] # 30组的极小值纵坐标
Globalfeatures = [[] for _ in range(data.shape[1])] # 预留位置30个
for i in range(data.shape[1]): # 对每组开始进行处理，循环30次
    temp = data.iloc[:, i] # 第i组
    '''全局特征'''
    mean = np.mean(temp)  # 均值
    std = np.std(temp)  # 标准差用于计算偏度和峭度
    var = np.var(temp)  # 方差
    skew = np.mean((temp - mean) ** 3) / pow(std, 3)  # 偏度
    kurt = np.mean((temp - mean) ** 4) / pow(var, 2)  # 峭度
    globalfeatures = [mean,var,skew,kurt] # 某一组的全局参数
    Globalfeatures[i] = globalfeatures # 所有组的全局参数

    '''极大值点'''
    max_max = signal.find_peaks(temp, distance=120)  # 设定极值之间的步长至少120
    xloc = max_max[0] # 返回极大值坐标和值 , 元组形式,这里[0]表示先考虑x坐标
    xlocnew = [[] for _ in range(len(xloc))]  # 用于存放筛选条件后的极大值点横坐标
    for j in range(len(xloc)):
        if xloc[j] <= 700 and xloc[j] > 80: #控制极大值点出现的范围，观察波形至少大于80，小于700
            xlocnew[j] = xloc[j]
        else:
            pass
    while [] in xlocnew: # xloc有很多极大值还需要筛选一下，筛选完后被淘汰的留下空值需要移除
        xlocnew.remove([])  # 移除空列表，此时已经得到了极大值点横坐标
    yloc = [[] for _ in range(len(xlocnew))] # 存放极大值点纵坐标
    for j in range(len(xlocnew)):
        yloc[j] = temp[xlocnew[j]]  # 极大值点纵坐标
    x_temp = [xlocnew[0],xlocnew[-1]] # 极大值只要2个,一般上述程序处理完剩3个
    y_temp = [yloc[0],yloc[-1]]
    #x_temp = np.array(x_temp)
    #y_temp = np.array(y_temp) # 只把第一个和最后一个作为两个极大值点,并归一化时间点处理
    x_max[i] = x_temp
    y_max[i] = y_temp

    '''极小值点'''
    min_min = signal.find_peaks(-1 * temp, distance=120)
    xloc = min_min[0]
    xlocnew = [[] for _ in range(len(xloc))]  # 用于存放筛选条件后的极小值点坐标
    for j in range(len(xloc)):
        if xloc[j] < 600 and xloc[j] > 400:
            xlocnew[j] = xloc[j]
        else:
            pass
    while [] in xlocnew:
        xlocnew.remove([])  # 移除空列表，此时已经得到了极小值点横坐标
    yloc = [[] for _ in range(len(xlocnew))]
    for j in range(len(xlocnew)):
        yloc[j] = temp[xlocnew[j]]  # 极小值点纵坐标

    m = min(yloc) # 极小值点只有1个，要最小的那个
    loction = [p for p, q in enumerate(yloc) if q == m]  # 记录最小值点中最小的那个的位置
    xx_temp = xlocnew[loction[0]]
    yy_temp = yloc[loction[0]]
    x_min[i] = xx_temp
    y_min[i] = yy_temp
k_x_max = pd.DataFrame(x_max) # 极大值点横坐标
k_x_max = k_x_max.div(1100)
Max_X = k_x_max.T
k_y_max = pd.DataFrame(y_max) #极大值点纵坐标
k_y_max = k_y_max.div(5.4)
Max_Y = k_y_max.T
k_x_min = pd.DataFrame(x_min) # 极小值点横坐标
k_x_min = k_x_min.div(1100)
Min_X = k_x_min.T
k_y_min = pd.DataFrame(y_min) # 极小值点纵坐标
k_y_min = k_y_min.div(5.4)
Min_Y = k_y_min.T
k_global =pd.DataFrame(Globalfeatures) # 全局特征
k_global = k_global
k_global.to_csv(kkadress+which+'.csv',index=None)
# 合并极大值点的横坐标和极小值点的横坐标
XX = pd.concat([Max_X,Min_X],axis=0,ignore_index= True)
XX = XX.T
# 合并极大值点的纵坐标和极小值点的纵坐标
YY = pd.concat([Max_Y,Min_Y],axis=0,ignore_index= True)
YY = YY.T
# 合并局部特征
Features = pd.concat([XX,YY],axis=1,ignore_index=True)
# 合并全局特征
Features = pd.concat([Features,k_global],axis=1,ignore_index=True)
# 行归一化处理
Features_values = Features.values
Features_values_normalized = Troubleshooting.dataprocessing.MaxMinNormalized(Features_values,mode=1,)
Features_normalized = pd.DataFrame(Features_values_normalized )
# 添加标签
#XX.to_csv(kadress+'时间_.csv',index=None)
#YY.to_csv(kadress+'幅值_.csv',index=None)
#Features_normalized.to_csv(kkadress+which+'.csv',index=None)
#%%
'''1、合并所有csv文件'''
address = "C:/Users/chenbei/Desktop/钢/线圈数据-20k/240/csv/"
Data = contact_csv(address)
Data_1 = [[] for _ in range(Data.shape[1])]
for i in range(Data.shape[1]) :
    data = Data.iloc[:, i]
    for j in range(len(data)):
        if (abs(data[j]) >= 0.2) and (abs(data[j+1] >= abs(data[j]))) and (abs(data[j]) <= 7):
            start = j - 2
            temp = data[start:(start + 1100)]
            temp1 = temp.values
            Data_1[i] = temp1.reshape(-1,1)
            break
Data_2 = pd.DataFrame()
for i in range(len(Data_1)):
    temp_data1 = pd.DataFrame(Data_1[i])
    Data_2 = pd.concat([Data_2,temp_data1],axis=1,ignore_index=True)
    #Data_2 = Data_2.div(5.4)
Data_2.to_csv(address+'data.csv',index=None)
for i in range((Data_2.shape[1])) :
    temp2 = Data_2.iloc[:,i]
    temp22 = temp2.values
    plt.plot(temp22)
plt.show()

#%%
'''2、文件批量命名'''
path = 'C:/Users/chenbei/Desktop/钢/线圈数据-20k/220kase/wdf/'
# C:/Users/chenbei/Desktop/钢/线圈数据-20k/180/wdf/
# C:/Users/chenbei/Desktop/钢/线圈数据-20k/220/wdf/
# C:\Users\chenbei\Desktop\钢\线圈数据-20k\220chuan10\wdf
files = os.listdir(path)
for i in range(len(files)) :
    oldname = path + files[i]
    newname = path+str(i)+ '.WDF'
    os.rename(oldname,newname)