#%%
# -*- coding UTF-8 -*-
'''
@Project : MyProjects
@File : getdata.py
@Author : chenbei
@Date : 2021/1/19 16:14
'''
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl
plt.rcParams['font.sans-serif'] = ['Times New Roman'] # 设置字体风格,必须在前然后设置显示中文
mpl.rcParams['font.size'] = 10.5 # 图片字体大小
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文的命令
mpl.rcParams['axes.unicode_minus'] = False  #  显示负号的命令
mpl.rcParams['agg.path.chunksize'] = 10000
plt.rcParams['figure.figsize'] = (7.8,3.8) # 设置figure_size尺寸
plt.rcParams['savefig.dpi'] = 600 # 图片像素
plt.rcParams['figure.dpi'] = 600 # 分辨率
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=10.5)
import numpy as np
import datetime
import tkinter as tk
from tkinter import *
from tkinter import ttk #各种界面组件
import tkinter.messagebox as tkMessageBox
from threading import Timer
import ctypes
from ctypes import *
import time

cModelAddr      = 0
cBufferLen        = 5120 # 缓冲长度
cAdChannel      = 8 # 通道个数

def ModelInit():
    tabwave.arr = []
    tkDemo.drv.model_init() # 调用模型初始化
    tkDemo.drv.model_rst()
    temp = c_ushort(0) # 无符号短整型
    p_temp = pointer(temp) # 指针指向
    tkDemo.drv.model_reg_read(cModelAddr,21,1,p_temp)
    print(p_temp[0])
    p_temp[0] = 0xAAAA #
    tkDemo.drv.model_reg_write(cModelAddr,21,1,p_temp)
    p_temp[0] = 0x0000
    tkDemo.drv.model_reg_read(cModelAddr,21,1,p_temp)
    print(p_temp[0])

    p_temp[0] = 0
    tkDemo.drv.model_reg_write(cModelAddr,9,1,p_temp)
    p_temp[0] = 4
    tkDemo.drv.model_reg_write(cModelAddr,3,1,p_temp)

    p_temp[0] = 1
    tkDemo.drv.model_reg_write(cModelAddr,23,1,p_temp)
    time.sleep(0.1)
    p_temp[0] = 0
    tkDemo.drv.model_reg_write(cModelAddr,23,1,p_temp)
    time.sleep(0.1)
    p_temp[0] = 1
    tkDemo.drv.model_reg_write(cModelAddr,23,1,p_temp)
    time.sleep(0.1)

def AdSet():
    tkDemo.arr = []
    temp = c_ushort(0)
    p_temp = pointer(temp)
    p_temp[0] = 0
    tkDemo.drv.model_reg_write(cModelAddr,10,1,p_temp)
    time.sleep(0.1)
    #cAdChannel  = 1
    p_temp[0]= 0xfffe #  254
    tkDemo.drv.model_reg_write(cModelAddr,0,1,p_temp)
    p_temp[0]= 0x0001#    1
    tkDemo.drv.model_reg_write(cModelAddr,1,1,p_temp)
    p_temp[0] = 0x0000 # 0
    tkDemo.drv.model_reg_write(cModelAddr,2,1,p_temp)
    p_temp[0] = 0x07d0   #ad fre is 1000
    tkDemo.drv.model_reg_write(cModelAddr,3,1,p_temp)
    p_temp[0] = 0x0300 # 768
    tkDemo.drv.model_reg_write(cModelAddr,9,1,p_temp)
    p_temp[0] = 0
    tkDemo.drv.model_reg_write(cModelAddr,11,1,p_temp)
    time.sleep(0.1)         #delay 0.1s

def AdStart():
    temp = c_ushort(0)
    p_temp = pointer(temp)
    p_temp[0] = 1
    tkDemo.drv.model_reg_write(cModelAddr,10,1,p_temp)
    time.sleep(0.1)

def GetData():
    AdStart()
    file = open('data.txt','a')
    while True :
        iCnt = tkDemo.drv.model_fifo_read_short(cModelAddr,19,cBufferLen,tkDemo.buf)
        for i in range(0,cBufferLen) :
            t1 = datetime.datetime.now()
            tkDemo.val[i] = tkDemo.buf[i]*10.0/0x8000 # 32768
            file.write(str(t1)+"||||"+str(tkDemo.val[i])+"\n")

if __name__ == '__main__':
    tkDemo = tk.Tk()
    tkDemo.title("Data Acquisition Demo") # 生成1个窗口
    tkDemo.geometry = ('200*200')

    tabCtrl = ttk.Notebook(tkDemo) # 窗口的1个菜单按钮

    tabwave = ttk.Frame(tabCtrl) # 波形框架继承界面
    tabCtrl.add(tabwave, text='Wave') # 添加文本 "波形"
    tabParam = ttk.Frame(tabCtrl)  # 添加文本 "参数"
    tabCtrl.add(tabParam, text='Param')
    tabCtrl.pack(expand=1, fill="both") # 容器放置在中央

    tabwave.tm = 0 # 波形的有两个参数
    tabwave.arr = [] #

    tkDemo.drv  = windll.LoadLibrary('./modelio.so') # 窗口熟悉drv加载了库
    tkDemo.buf  = (c_short*cBufferLen)() # 缓存长度 short型
    tkDemo.val  = (c_float*cBufferLen)() #  存放数据的
    tkDemo.view = (c_int*cBufferLen)() # int 型
    tkDemo.rdVals = (c_int*cBufferLen)()

    ModelInit()
    AdSet()

    frame = Frame(tkDemo) # 框架部件继承
    frame.pack() # 框架居于中央

    btnStart = Button(frame, text="Start", command = GetData()) # 定义两个按钮 分别是获取数据和启动采集
    btnClose = Button(frame, text="Close", command = AdSet())
    btnStart.grid(row = 1, column = 5) # 按钮的长度和宽度
    btnClose.grid(row = 1, column = 10)

    tkDemo.mainloop()
