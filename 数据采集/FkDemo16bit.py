# -*- coding: utf-8 -*-
import tkinter as tk
from tkinter import *
from tkinter import ttk
import tkinter.messagebox as tkmessageBox
from threading import Timer
import ctypes
from ctypes import *
import time
import re

cCanvasWidth = 640 # 画布宽度
cCanvasHeight = 480 # 高度
cCanvasGrid = 10 # 画布网格
cTimePeriod = 1 # 周期

cModelAddr      = 0
cBufferLen        = 10240 # 缓冲长度
cAdChannel      = 16 # 通道个数
cViewChannel =  0 # 所在通道

def TimerTask():
    DispClear()
    DispWave()
    tabwave.tm.cancel()
    tabwave.tm = Timer(cTimePeriod, TimerTask)
    tabwave.tm.start()
def DispClear():
    tabwave.canvas.delete('wave')
def DispWave():
    horizonPos = cCanvasHeight/2
    chadnum = cBufferLen/cAdChannel
    #print("model chadnum is %D\n",chadnum);
    LoadData()
    for i in range(0, chadnum-1):
        line = tabwave.canvas.create_line(i, (tabwave.arr[i*cAdChannel+cViewChannel] + horizonPos), (i+1), (tabwave.arr[(i+1)*cAdChannel+cViewChannel] + horizonPos), fill='lime', width=1,tag='wave')
def StartTimerTask():
    tabwave.tm = Timer(cTimePeriod, TimerTask)
    tabwave.tm.start()

def StopTimerTask():
    if tabwave.tm != 0:
        tabwave.tm.cancel()

def DrawGrid():
    line = tabwave.canvas.create_line(0, (cCanvasHeight/2), cCanvasWidth, (cCanvasHeight/2), fill='lightgray', width=1, tag='grid')
    line = tabwave.canvas.create_line((cCanvasWidth/2), 0, (cCanvasWidth/2), cCanvasHeight, fill='lightgray', width=1, tag='grid')

    cnt = cCanvasHeight/2/cCanvasGrid
    for i in range(1, cnt):
        x = i*cnt + cCanvasWidth/2
        line = tabwave.canvas.create_line(x, 0, x, cCanvasHeight, fill='dimgray', width=1, dash=(4, 4), tag='grid')
        x = cCanvasWidth/2 - i*cnt
        line = tabwave.canvas.create_line(x, 0, x, cCanvasHeight, fill='dimgray', width=1, dash=(4, 4), tag='grid')

    cnt = cCanvasWidth/2/cCanvasGrid
    for i in range(1, cnt):
        y = i*cnt + cCanvasHeight/2
        line = tabwave.canvas.create_line(0, y, cCanvasWidth, y, fill='dimgray', width=1, dash=(4, 4), tag='grid')
        y = cCanvasHeight/2 - i*cnt
        line = tabwave.canvas.create_line(0, y, cCanvasWidth, y, fill='dimgray', width=1, dash=(4, 4), tag='grid')

def ModelInit():
    tabwave.arr = []
    tkDemo.drv.model_init()
    tkDemo.drv.model_rst()
    temp = c_ushort(0)
    p_temp = pointer(temp)
    tkDemo.drv.model_reg_read(cModelAddr,21,1,p_temp) #数据总线校验测试。可以写一个数，立即读取，看是否一致。
    print(p_temp[0])
    p_temp[0] = 0xAAAA # 1010 1010 1010 1010
    tkDemo.drv.model_reg_write(cModelAddr,21,1,p_temp)
    p_temp[0] = 0x0000 # 数据总线校验测试。可以写一个数，立即读取，看是否一致。
    tkDemo.drv.model_reg_read(cModelAddr,21,1,p_temp) # 写了个0，读取观察输出是否是0
    print(p_temp[0])

    p_temp[0] = 0 # 其实就是0000 0000 0000 0000 初始化 默认是读取±5V
    tkDemo.drv.model_reg_write(cModelAddr,9,1,p_temp)

    p_temp[0] = 4 # 0000 0000 0000 0100
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
    temp = c_ushort(0) # 无符号整型0~65535 # 定义了一个无符号整型数0，为了和C的函数匹配
    p_temp = pointer(temp) # ctypes中使用POINTER和pointer表示指针，在使用POINTER时需要设置指向的数据类型，而pointer则直接从变量中得到一个特定类型的指针

    p_temp[0] = 0 # p_temp可以有很多值,如果 H = p_temp[0] 单独取出来就是一个整型int
    tkDemo.drv.model_reg_write(cModelAddr,10,1,p_temp) #模型地址、寄存器地址、寄存器编号、寄存器数据
    #寄存器10是启动采集，数据写1，需要延时 100ms 才能读寄存器 19 的采集数据，延时并不影响采集的实时性
    #第 10 寄存器写 0，停止采集。设置参数之前，必须停止采集，否则可能造成模块间的同步误差，这里设置p_temp即0
    time.sleep(0.1) # 延时100ms

    #cAdChannel  = 1
    p_temp[0] = 0xfffe # 1111  1111  1111 1110 通道掩码，使能通道的起始通道位置 0，其他所有通道位置为1
    tkDemo.drv.model_reg_write(cModelAddr,0,1,p_temp)

    p_temp[0] = 0x0001#通道使能，该值 16bit，分别对应 Ain00-15 通道
    #的使能和禁止，建议使能的通道之间应该连续。
    tkDemo.drv.model_reg_write(cModelAddr,1,1,p_temp)

    p_temp[0] = 0x0000 #采样频率分频系数 24bit 的高 8bit
    tkDemo.drv.model_reg_write(cModelAddr,2,1,p_temp)

    p_temp[0] = 0x07d0 #采样频率分频系数 24bit 的低 16bit      #ad fre is 1000 分频系数2000
    tkDemo.drv.model_reg_write(cModelAddr,3,1,p_temp)

    p_temp[0] = 0x0300 # 0000 0011 0000 0000 从右边读，11表示两个10v，第8,9位
    tkDemo.drv.model_reg_write(cModelAddr,9,1,p_temp)
    #选择时钟为基准时钟。即把每个需要同步的 SK3101 模块第 09 寄存器的Bit0=1
    #Bit0：内部时钟源选择，=0 为该模块板载时钟，=1 为底板系统基准时钟，一般在多个板卡同步采集时使用；
    #Bit1：秒同步使能选择，=0 禁止，=1 使能，暂时不用；
    #Bit2：触发管脚信号输出类型选择（Bit3=1 时有效），=0 为该模块设置的触发控制信号状态输出，=1 为 A/D 转换时钟输出；
    #Bit3：触发管脚信号输入输出方向选择，=0 为输入，=1 为输出；
    #Bit4-7：暂时不用，置 0；
    #Ain00-07 通道的输入范围，=0 为±5V，=1 为±10V；
    #Bit9： Ain08-15 通道的输入范围，=0 为±5V，=1 为±10V；
    #Bit10：暂时不用，置 0；
    #通道使能设置后，参与实际采集通道数，数据范围 1-16；

    p_temp[0] = 0
    tkDemo.drv.model_reg_write(cModelAddr,11,1,p_temp)
    time.sleep(0.1)         #delay 0.1s

def AdStart():
    temp = c_ushort(0)
    p_temp = pointer(temp)
    p_temp[0] = 1  
    tkDemo.drv.model_reg_write(cModelAddr,10,1,p_temp)
    time.sleep(0.1)

def LoadData(): 
    tabwave.arr = []
    iCnt = tkDemo.drv.model_fifo_read_short(cModelAddr, 19, cBufferLen, tkDemo.buf)
    for i in range(0, cBufferLen):
	tkDemo.val[i]  = tkDemo.buf[i]*10.0/ 0x8000
	tkDemo.view[i] = tkDemo.buf[i]*(cCanvasHeight/2)/0x8000 
    #tkDemo.buf[i] = tkDemo.buf[i]/100
	tabwave.arr.append(tkDemo.view[i])


def CloseMe():
    StopTimerTask()
    tkDemo.drv.model_close()
    tkDemo.quit()

def StartOneTask():
    DispClear()
    DispWave()

def CloseOneTask():
    tkDemo.drv.model_close()
    tkDemo.quit()

def ReadValue():
    modAddr = int(rdModSel.get())
    regAddr = int(rdRegSel.get())
    tkDemo.drv.model_reg_read(modAddr, regAddr, 1, tkDemo.rdVals)
    txtOut.set('%04X'%tkDemo.rdVals[0])

def WriteValue():
    modAddr = int(rdModSel.get())
    regAddr = int(rdRegSel.get())
    if(txtIn.get() != ""):
        try:
            tkDemo.rdVals[0] = int(txtIn.get(), 16)
            tkDemo.drv.model_reg_write(modAddr, regAddr,1, tkDemo.rdVals)
        except:
            tkmessageBox.showinfo('Warning','Please input a hex string!')

if __name__ == '__main__':
    tkDemo = tk.Tk()
    tkDemo.title("Data Acquisition Demo")
    tabCtrl = ttk.Notebook(tkDemo)

    tabwave = ttk.Frame(tabCtrl)
    tabCtrl.add(tabwave, text='Wave')
    tabParam = ttk.Frame(tabCtrl)
    tabCtrl.add(tabParam, text='Param')

    tabCtrl.pack(expand=1, fill="both")

    tabwave.tm = 0
    tabwave.arr = []

    tkDemo.drv  = cdll.LoadLibrary('./modelio.so')
    tkDemo.buf  = (c_short*cBufferLen)()
    tkDemo.val  = (c_float*cBufferLen)()    
    tkDemo.view = (c_int*cBufferLen)()
    tkDemo.rdVals = (c_int*cBufferLen)()

    ModelInit()
    #time.sleep(5)
    AdSet()
    #time.sleep(5)
    AdStart()

    tabwave.canvas = Canvas(tabwave, width = cCanvasWidth, height = cCanvasHeight, bg = "black")
    tabwave.canvas.pack()

    frame = Frame(tkDemo)
    frame.pack()

    btnStart = Button(frame, text="Start", command = StartTimerTask)
    btnClose = Button(frame, text="Close", command = CloseMe)
    btnStartOne = Button(frame, text="StartOne", command = StartOneTask)
    btnCloseOne = Button(frame, text="CloseOne", command = CloseOneTask)

    btnStart.grid(row = 1, column = 1)
    btnClose.grid(row = 1, column = 2)
    btnStartOne.grid(row = 1, column = 3)
    btnCloseOne.grid(row = 1, column = 4)

    DrawGrid()

    RdFrm = ttk.LabelFrame(tabParam, text='Read Parameter')
    RdFrm.grid(column=0, row=0, padx=8, pady=4)

    lblMA = ttk.Label(RdFrm, text="Module Address")
    lblMA.grid(column=0, row=0, sticky='W')

    lblSP = ttk.Label(RdFrm, text="  ")
    lblSP.grid(column=1, row=0, sticky='W')

    lblRA = ttk.Label(RdFrm, text="Register Address")
    lblRA.grid(column=2, row=0, sticky='W')

    lblRV = ttk.Label(RdFrm, text="Read value(Hex)")
    lblRV.grid(column=0, row=3, sticky='W')

    lblSP = ttk.Label(RdFrm, text="")
    lblSP.grid(column=0, row=5, sticky='W')

    rdMod = tk.StringVar()
    rdModSel = ttk.Combobox(RdFrm, textvariable=rdMod, state='readonly')
    rdModSel['values'] = range(0,16)
    rdModSel.grid(column=0, row=1)
    rdModSel.current(0)

    rdReg = tk.StringVar()
    rdRegSel = ttk.Combobox(RdFrm, textvariable=rdReg, state='readonly')
    rdRegSel['values'] = range(0,32)
    rdRegSel.grid(column=2, row=1)
    rdRegSel.current(0)

    txtOut = tk.StringVar()
    txtRd = ttk.Entry(RdFrm, width = 22, textvariable=txtOut, state='readonly')
    txtRd.grid(column=0, row=4, sticky='W')

    btnRd = ttk.Button(RdFrm, width = 22, text="Read", command = ReadValue)
    btnRd.grid(column=2, row=4, sticky='W')

    WrFrm = ttk.LabelFrame(tabParam, text='Write Parameter')
    WrFrm.grid(column=0, row=0+6, padx=8, pady=4)

    lblMA = ttk.Label(WrFrm, text="Module Address")
    lblMA.grid(column=0, row=0+6, sticky='W')

    lblSP = ttk.Label(WrFrm, text="  ")
    lblSP.grid(column=1, row=0+6, sticky='W')

    lblRA = ttk.Label(WrFrm, text="Register Address")
    lblRA.grid(column=2, row=0+6, sticky='W')

    lblRV = ttk.Label(WrFrm, text="Write value(Hex)")
    lblRV.grid(column=0, row=3+6, sticky='W')

    lblSP = ttk.Label(WrFrm, text="")
    lblSP.grid(column=0, row=5+6, sticky='W')

    wrMod = tk.StringVar()
    wrModSel = ttk.Combobox(WrFrm, textvariable=wrMod, state='readonly')
    wrModSel['values'] = range(0,16)
    wrModSel.grid(column=0, row=1+6)
    wrModSel.current(0)

    wrReg = tk.StringVar()
    wrRegSel = ttk.Combobox(WrFrm, textvariable=wrReg, state='readonly')
    wrRegSel['values'] = range(0,32)
    wrRegSel.grid(column=2, row=1+6)
    wrRegSel.current(0)

    txtIn = tk.StringVar()
    txtWr = ttk.Entry(WrFrm, width = 22, textvariable=txtIn)
    txtWr.grid(column=0, row=4+6, sticky='W')

    btnWr = ttk.Button(WrFrm, width = 22, text="Write", command = WriteValue)
    btnWr.grid(column=2, row=4+6, sticky='W')

    tkDemo.resizable(0,0)
    tkDemo.protocol("WM_DELETE_WINDOW", CloseMe)
    tkDemo.mainloop()
