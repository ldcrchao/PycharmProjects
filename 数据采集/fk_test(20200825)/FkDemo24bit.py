# -*- coding: utf-8 -*-
import Tkinter as tk
from Tkinter import *
import ttk
import tkMessageBox

from threading import Timer

import ctypes
from ctypes import *
import time

import re

cCanvasWidth = 640
cCanvasHeight = 480
cCanvasGrid = 10
cTimePeriod = 1

cModelAddr      = 0
cBufferLen        = 10240
cAdChannel      = 16
cViewChannel =  0

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
    tkDemo.drv.model_reg_read(cModelAddr,21,1,p_temp)
    print(p_temp[0])
    p_temp[0] = 0xAAAA
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
    p_temp[0] = 0xffff
    tkDemo.drv.model_reg_write(cModelAddr,1,1,p_temp)
    p_temp[0] = 256          #ad fre is 1000 
    tkDemo.drv.model_reg_write(cModelAddr,3,1,p_temp)
    p_temp[0] = 0x8900 
    tkDemo.drv.model_reg_write(cModelAddr,9,1,p_temp)
    p_temp[0] = 0 
    tkDemo.drv.model_reg_write(cModelAddr,11,1,p_temp)
    time.sleep(0.01)         #delay 0.01s
    p_temp[0] = 1 
    tkDemo.drv.model_reg_write(cModelAddr,11,1,p_temp)
    time.sleep(0.01)
    p_temp[0] = 0  
    tkDemo.drv.model_reg_write(cModelAddr,11,1,p_temp)
    time.sleep(0.01)
    p_temp[0] = 1  
    tkDemo.drv.model_reg_write(cModelAddr,11,1,p_temp)
    time.sleep(0.01)

def AdStart():
    temp = c_ushort(0)
    p_temp = pointer(temp)
    p_temp[0] = 1  
    tkDemo.drv.model_reg_write(cModelAddr,10,1,p_temp)
    time.sleep(0.1)

def LoadData(): 
    tabwave.arr = []
    iCnt = tkDemo.drv.model_fifo_read_int(cModelAddr, 19, cBufferLen, tkDemo.buf)
    for i in range(0, cBufferLen):
	tkDemo.val[i]     = tkDemo.buf[i]*10.0/0x80000000
	tkDemo.view[i] = tkDemo.buf[i]*(cCanvasHeight/2)/0x80000000 
        #tkDemo.buf[i] = tkDemo.buf[i]/100
	tabwave.arr.append(tkDemo.view[i])

def DispWave():
    horizonPos = cCanvasHeight/2
    chadnum = cBufferLen/cAdChannel
    #print("model chadnum is %D\n",chadnum);
    LoadData()
    for i in range(0, chadnum-1):
        line = tabwave.canvas.create_line(i, (tabwave.arr[i*cAdChannel+cViewChannel] + horizonPos), (i+1), (tabwave.arr[(i+1)*cAdChannel+cViewChannel] + horizonPos), fill='lime', width=1,tag='wave')

def DispClear():
    tabwave.canvas.delete('wave')

def TimerTask():
    DispClear()
    DispWave()
    tabwave.tm.cancel()
    tabwave.tm = Timer(cTimePeriod, TimerTask)
    tabwave.tm.start()

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
            tkMessageBox.showinfo('Warning','Please input a hex string!')

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
    tkDemo.buf  = (c_int*cBufferLen)()
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
