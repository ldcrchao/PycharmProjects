#%%
import pandas as pd
import numpy as np
from faker import Faker
fake = Faker(locale='zh_CN') #美国英文：en_US , 英国英文：en_GB , 简体中文：zh_CN , 繁体中文：zh_TW
name = fake.name()
address = fake.address()
print(fake.random_int(min=1 ,max=100)) #随机整数
print(fake.random_digit()) # 0~9随机数
print(fake.random_number(digits=5))
print(fake.pyfloat(left_digits=2,right_digits=3,positive=True)) #随机2位整数，3位小数的正数
print(fake.pydecimal(left_digits=2,right_digits=3,positive=False))
#%%
import socket
import datetime
import time
import pandas as pd
import datetime
import pywt
import numpy as np
import matplotlib.pyplot as plt
import math
from time import sleep
from tqdm import tqdm
from pandas import DataFrame,Series
from Car import Troubleshooting
#%%
'''调用包的测试程序'''
diabetes = pd.read_csv('C:/Users\chenbei\Documents\python数据\pycaret-master\datasets\diabetes.csv')
self_1 = Troubleshooting.feature_predict(diabetes) #实例化故障预测类
clf = self_1.SetUp(target='Class variable')
best = self_1.CompareModels()
print(best)
LR = self_1.CreateModels(estimator='lr')
LR_calibrate = self_1.CalibrationModels(LR)
'''model是已经窗创建好的模型，可以带入createmodel或者calibrationmodel得到的模型，dataframe是新数据'''
pre_model , Pre = self_1.PredictModels(model=LR,dataframe=diabetes)  # 返回模型的得分和预测标签 / 和返回新数据预测的得分和标签
self_1.SaveModels(pre_model,'最终预测模型')
#%%
x0 = np.array(np.random.randn(10, 5))
#self_dataprocess = Troubleshooting.dataprocessing(x0) #实例化数据处理类
norm_col = Troubleshooting.dataprocessing.MaxMinNormalized(x0,0) #按列归一化
norm_col_reverse = Troubleshooting.dataprocessing.MaxMinNormalized(norm_col,0,reverse=True ,col_max=x0.max(axis=0),col_min=x0.min(axis=0)) #按列反归一化
#%%
tcpClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 客户端
tcpClient.connect(('192.168.1.198', 1600))
self_tcp = Troubleshooting.tcp(tcpClient) #实例化tcp通信类
'''设置采集命令'''
Settings = self_tcp.ADset()   # 成功时返回值 = 1
'''启动采集命令'''
Start = self_tcp.ADstart()   # 成功时返回值 = 1
sample_time_start = datetime.datetime.now()
print("数据采集开始", sample_time_start)
sample_time_end = sample_time_start + datetime.timedelta(seconds=10)
print("数据采集计划结束", sample_time_end)
DATA_tuple = []  # 16进制元组
# DATA_str = []   #16进制字符串
DATA_signed10 = []  # 有符号十进制数
while True:
    '''数据接收'''
    temp_data = self_tcp.ADDataRead()  # 1440个字节
    '''数据处理'''
    dt, dict_10, dict_tmp = self_tcp.csv_tuple_str(temp_data)
    DATA_tuple.append(dict_tmp)
    # DATA_str.append(Dict_tmp)
    DATA_signed10.append(dict_10)
    time_now = datetime.datetime.now()
    if dt > sample_time_end:
        print("数据存储结束", time_now)
        break
    df_tuple = pd.DataFrame(DATA_tuple)
    # df_str = pd.DataFrame(DATA_str)
    df_signed_10 = pd.DataFrame(DATA_signed10)
    df_tuple.to_csv("C:/Users\chenbei\Desktop\钢\data_tuple.csv", index=False)  # 不保存标签
    # df_str.to_csv("C:/Users\chenbei\Desktop\钢\data_str.csv", index=False)
    df_signed_10.to_csv("C:/Users\chenbei\Desktop\钢\data_signed_10.csv", index=False)
self_tcp.tcpClient.close()
#%%
def ProgressBar(start=None, mode=False):
    '''mode默认为假，执行下属程序；如果mode=True只输出:执行开始'''
    try:
        scale = 50
        if not mode:  # mode为假或者默认会执行下属程序，如果mode指定True则只执行第一句
            # print("执行开始，祈祷不报错".center(scale // 2, "-"))
            # start = time.perf_counter()
            for i in range(scale + 1):
                a = "*" * i
                b = "." * (scale - i)
                c = (i / scale) * 100
                dur = time.perf_counter() - start
                print("\r{:^3.0f}%[{}->{}]{:.2f}s".format(c, a, b, dur), end="")
                time.sleep(0.1)
            print("\n" + "执行结束，万幸".center(scale // 2, "-"))
        else:
            print("执行开始，祈祷不报错".center(scale // 2, "-"))
    except:
        print('不好意思，程序有BUG，您需要重新调试！')

start = time.perf_counter()
ProgressBar(mode=True) #只返回第一句的提示信息
data1 = pd.read_excel("C:/Users\chenbei\Desktop\钢\分合闸储能波形\分合闸储能波形\储能.xlsx")
data2 = pd.read_excel("C:/Users\chenbei\Desktop\钢\分合闸储能波形\分合闸储能波形\分闸.xlsx")
data3 = pd.read_excel("C:/Users\chenbei\Desktop\钢\分合闸储能波形\分合闸储能波形\合闸.xlsx")
del data1['采样点']
del data2['采样点']
del data3['采样点']
# 重采样处理
timenow = datetime.datetime.now()
Index1 = pd.date_range(timenow, periods=len(data1), freq='S')
Index2 = pd.date_range(timenow, periods=len(data2), freq='S')
Index3 = pd.date_range(timenow, periods=len(data3), freq='S')
data1.index = Index1
data2.index = Index2
data3.index = Index3
Data1 = data1.resample('100S').sum()
Data2 = data2.resample('100S').sum()
Data3 = data3.resample('100S').sum()
import matplotlib.pyplot as plt
import numpy as np
T1 = Data1.values / abs(np.max(Data1.values))
T2 = Data2.values / abs(np.max(Data2.values))
T3 = Data3.values / abs(np.max(Data3.values))
#%%
# 观察波形
plt.subplot(3, 1, 1)
plt.plot(T1)
plt.subplot(3, 1, 2)
plt.plot(T2)
plt.subplot(3, 1, 3)
plt.plot(T2)
plt.show()

# 截取并重命名
D1 = Data1[700:2000]
D2 = Data2[300:1600]
D3 = Data3[300:1600]
D1.rename(columns={"储能电流值/A": "values"}, inplace=True)
D1.index = np.arange(0, D1.shape[0], 1)  # 步长
D2.rename(columns={"分闸电流值/A": "values"}, inplace=True)
D2.index = np.arange(0, D2.shape[0], 1)  # 步长
D3.rename(columns={"合闸电流值/A": "values"}, inplace=True)
D3.index = np.arange(0, D3.shape[0], 1)  # 步长
Data = pd.concat([D1, D2, D3], axis=0, ignore_index=False)
Data['time'] = Data.index
Data[['values', 'time']] = Data[['time', 'values']]
Data.columns = ['time', 'values']
# 拼接法得到id列
A1 = np.ones(D1.shape[0], np.int).reshape(-1, 1)
A2 = 2 * np.ones(D1.shape[0], np.int).reshape(-1, 1)
A3 = 3 * np.ones(D3.shape[0], np.int).reshape(-1, 1)
A = np.concatenate((A1, A2, A3))
Data.insert(0, 'id', A)
# 保存数据
# Data.to_csv("C:/Users\chenbei\Desktop\钢\分合闸储能波形\分合闸储能波形分合闸储能重采样合并数据.csv",index=False)
'缺失值调试'
#self0 = Troubleshooting.feature_extract(Data1.values)
# kkk = self0.WaveletAlternation(Data1.values) # 说明输入时间序列必须是按列的，每列是时间序列
# miss = self0.draw_missing_data_table(Data1)
D = pd.concat([D1, D2, D3], axis=1, ignore_index=True)
self1 = Troubleshooting.feature_extract(D.values)
test_D = self1.WaveletAlternation(D.values)
#%%
ecg = pywt.data.ecg() #引入测试数据
plt.plot(ecg)
plt.show()
pythonquzao = Troubleshooting.dataprocessing.Python_Quzao(ecg)
plt.plot (pythonquzao)
plt.show()
matlabquzao = Troubleshooting.dataprocessing.Matlab_Quzao(ecg)
plt.plot (matlabquzao)
plt.show()
ProgressBar(start) #这里需要给出初始时间
#%%
plt.plot(data3.values[119000:119400])
plt.show()
kkk = Troubleshooting.dataprocessing.Mean_3_5(data3.values[119000:119400],3)
#%%
D.to_csv("C:/Users\chenbei\Desktop\钢\D.csv",index=False)













