# -*- coding: utf-8 -*-
from Tkinter import *
from threading import Timer

import ctypes
from ctypes import *
import time

cCanvasWidth = 640
cCanvasHeight = 480
cCanvasGrid = 10
cTimePeriod = 1

cModelAddr = 0
cBufferLen   =  10240
cAdChannel  = 16

#Data Acquisition Demo
tkDemo = Tk()

def TimerTask():
    DispClear()
    DispWave()
    tkDemo.tm.cancel()
    tkDemo.tm = Timer(cTimePeriod, TimerTask)
    tkDemo.tm.start()

def StartTimerTask():
    tkDemo.tm = Timer(cTimePeriod, TimerTask)
    tkDemo.tm.start()

def StopTimerTask():
    if tkDemo.tm != 0:
        tkDemo.tm.cancel()

def DrawGrid():
    line = tkDemo.canvas.create_line(0, (cCanvasHeight/2), cCanvasWidth, (cCanvasHeight/2), fill='lightgray', width=1, tag='grid')
    line = tkDemo.canvas.create_line((cCanvasWidth/2), 0, (cCanvasWidth/2), cCanvasHeight, fill='lightgray', width=1, tag='grid')

    cnt = cCanvasHeight/2/cCanvasGrid
    for i in range(1, cnt):
        x = i*cnt + cCanvasWidth/2
        line = tkDemo.canvas.create_line(x, 0, x, cCanvasHeight, fill='dimgray', width=1, dash=(4, 4), tag='grid')
        x = cCanvasWidth/2 - i*cnt
        line = tkDemo.canvas.create_line(x, 0, x, cCanvasHeight, fill='dimgray', width=1, dash=(4, 4), tag='grid')

    cnt = cCanvasWidth/2/cCanvasGrid
    for i in range(1, cnt):
        y = i*cnt + cCanvasHeight/2
        line = tkDemo.canvas.create_line(0, y, cCanvasWidth, y, fill='dimgray', width=1, dash=(4, 4), tag='grid')
        y = cCanvasHeight/2 - i*cnt
        line = tkDemo.canvas.create_line(0, y, cCanvasWidth, y, fill='dimgray', width=1, dash=(4, 4), tag='grid')

def ModelInit():
    tkDemo.arr = []
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
    p_temp[0] = 0x8901 
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
    tkDemo.arr = []
    iCnt = tkDemo.drv.model_fifo_read_int(cModelAddr, 19, cBufferLen, tkDemo.buf)
    for i in range(0, cBufferLen):
	tkDemo.val[i]     = tkDemo.buf[i]*10.0/0x80000000
	tkDemo.view[i] = tkDemo.buf[i]*(cCanvasHeight/2)/0x80000000 
        #tkDemo.buf[i] = tkDemo.buf[i]/100
	tkDemo.arr.append(tkDemo.view[i])

def DispWave():
    horizonPos = cCanvasHeight/2
    chadnum = cBufferLen/cAdChannel
    #print("model chadnum is %D\n",chadnum);
    LoadData()
    for i in range(0, chadnum-1):
        line = tkDemo.canvas.create_line(i, (tkDemo.arr[i*cAdChannel] + horizonPos), (i+1), (tkDemo.arr[(i+1)*cAdChannel] + horizonPos), fill='lime', width=1,tag='wave')

def DispClear():
    tkDemo.canvas.delete('wave')



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

if __name__ == '__main__':
    tkDemo.title("Data Acquisition Demo")
    tkDemo.tm = 0
    tkDemo.arr = []

    tkDemo.drv  = cdll.LoadLibrary('./modelio.so')
    tkDemo.buf  = (c_int*cBufferLen)()
    tkDemo.val  = (c_float*cBufferLen)()    
    tkDemo.view = (c_int*cBufferLen)()

    ModelInit()
    #time.sleep(5)
    AdSet()
    #time.sleep(5)
    AdStart()

    tkDemo.canvas = Canvas(tkDemo, width = cCanvasWidth, height = cCanvasHeight, bg = "black")
    tkDemo.canvas.pack()

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

    tkDemo.resizable(0,0)
    tkDemo.protocol("WM_DELETE_WINDOW", CloseMe)
    tkDemo.mainloop()
