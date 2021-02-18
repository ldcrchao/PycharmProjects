#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : 第一章的饼图.py
@Author : chenbei
@Date : 2020/12/19 16:35
'''
from matplotlib.pylab import mpl
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman'] # 设置字体风格,必须在前然后设置显示中文
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文的命令
mpl.rcParams['axes.unicode_minus'] = False  #  显示负号的命令
plt.rcParams['figure.figsize'] = (7.8,4.8) # 设置figure_size尺寸
plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
plt.rcParams['savefig.dpi'] = 600 # 图片像素
plt.rcParams['figure.dpi'] = 600 # 分辨率
#%%
labels = ['施耐德', '正泰电器', 'ABB', '德力西', '西门子','常熟开关','良信电器','上海人民电器','其它']  # 自定义标签
sizes = [15, 15, 6, 5, 3,3,2,2,49]  # 每个标签占多大
explode = (0, 0, 0, 0, 0,0,0,0,0)  # 将某部分爆炸出来
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)
plt.axis('equal')
plt.title('2018年国内LVCB整体市场份额分布')
plt.show()
#参数autopct，展示比数值，可取值：
#a. %d%%：整数百分比;
#b. %0.1f：一位小数；
#c. %0.1f%%：一位小数百分比；
#d. %0.2f%%：两位小数百分比
#%%
labels = ['配电电器','终端电器','控制电器','其它']
sizes = [35,33,18,14]
explode = [0 , 0 ,0 ,0]
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)
plt.axis('equal')
plt.title('2018年不同用途的低压断路器产值占比图')
plt.show()
#%%
labels = ['微型断路器','塑壳断路器','接触器','框架断路器','其它']
sizes = [32,23,19,19,7]
explode = [0 , 0 ,0 ,0,0]
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)
plt.axis('equal')
plt.title('2018年不同结构的低压断路器产值占比图')
plt.show()
#%%
labels = ['全部定期检修','维修周期开始调整','部分状态检修','全部状态检修']
sizes = [26,37,27.2,9.8]
explode = [0 , 0 ,0 ,0]
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=90)
plt.axis('equal')
plt.title('国内34家供电公司设备不同检修方式的占比')
plt.show()
#%%
labels = ['电力','建筑','工控','石化','电信数据中心','新能源','交通','其它']
height = [0.4 , 0.28,0.07,0.07,0.03,0.02,0.02,0.11]
reats = plt.bar(labels,height,width = 0.35,align='center',color = 'c',alpha=0.8)
for reat in reats : # 每一个柱子循环标注数值大小
    reat_height = reat.get_height() # 获取柱子高度
    plt.text(reat.get_x() + reat.get_width()/2,reat_height,str(reat_height),size=10.5,ha='center',va='bottom')
plt.tight_layout()
plt.title('低压断路器主要应用的行业和需求占比')
plt.show()
#%%
labels = ['2009年','2010年','2011年','2012年','2013年','2014年','2015年','2016年','2017年','2018年']
height = [440 ,500,560,610,680,730,670,678,781,841]
reats = plt.bar(labels,height,width=0.35,align='center',alpha=0.8)
for reat in reats : # 每一个柱子循环标注数值大小
    reat_height = reat.get_height() # 获取柱子高度
    plt.text(reat.get_x() + reat.get_width()/2,reat_height,str(reat_height),size=10.5,ha='center',va='bottom')
plt.tight_layout()
plt.title('中国低压电器工业总产值(亿元)')
plt.show()

