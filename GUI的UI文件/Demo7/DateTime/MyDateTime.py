#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : MyDateTime.py
@Author : chenbei
@Date : 2020/10/5 10:31
'''
import sys
from PyQt5.QtWidgets import  QApplication,QWidget
from PyQt5.QtCore import  QDate ,QDateTime ,QTime
from GUI.Demo.Demo5 import QDateTime as Q
from Car.Troubleshooting import  dataprocessing as D
class QMyDateTime(QWidget):
    '''
    读取时间要使用完整格式yyyy-MM-dd hh:mm:ss:zzz
    显示时间可以根据需要显示
    '''
    def __init__(self,parent=None):
        super(QMyDateTime, self).__init__(parent)
        self.ui = Q.Ui_QDateTime()
        self.ui.setupUi(self)
        self.ui.settimepushButton.setDefault(True) # 可以突出显示按钮框
        self.ui.setdatepushButton.setDefault(True)
        self.ui.setdatetimepushButton.setDefault(True)
        self.ui.readdatetime.setDefault(True)
        self.show()
        self.ui.settimepushButton.clicked.connect(self.on_settimepushButton_clicked)
        self.ui.setdatepushButton.clicked.connect(self.on_setdatepushButton_clicked)
        self.ui.setdatetimepushButton.clicked.connect(self.on_setdatetimepushButton_clicked)
    def on_readdatetime_clicked(self):
        '''读取当前时间显示'''
        '''关联3个日期组件'''
        time = QDateTime.currentDateTime()
        self.ui.timeEdit.setTime(time.time()) # 提出时间给时间组件
        self.ui.dateEdit.setDate(time.date()) # 提出日期给日期组件
        #print(time.date())
        self.ui.dateTimeEdit.setDateTime(time) # 全部提取
        '''关联3个字符串显示文本'''
        self.ui.timestrshow.setText(time.toString("hh:mm:ss"))# 提取时间转换成指定格式的字符串赋给显示文本
        #不使用mm可能17:00:59显示成17:0_:59
        self.ui.timestrshow_2.setText(time.toString("yyyy-MM-dd"))
        self.ui.timestrshow_3.setText(time.toString("yyyy-MM-dd hh:mm:ss:zzz"))
    '''读取文本组件设定对应的日期or时间组件or日期时间组件进行显示'''
    def on_settimepushButton_clicked(self):
        '''读取界面文本组件(人工输入)在时间组件进行显示'''
        str = self.ui.timestrshow.text()
        # 采用解决方法1,格式对应即可
        tmstr = QTime.fromString(str,"hh时mm分ss秒") # 读取指定格式的文本进行显示
        #为了能够让人工输入指定格式的文本,在设计之初就规定了相应格式的属性,可见窗体文件ui inputMask的格式
        # 主要含义如下
        # ------------------------------------------------------
        # 99:99:99;_              | 只能输入0-9的数字,空格用"_"代替
        # 9999-99-99              | 只能输入0-9的数字
        # 9999-99-99 99:99:99:999 | 只能输入0-9的数字
        # ------------------------------------------------------
        self.ui.timeEdit.setTime(tmstr)
    def on_setdatepushButton_clicked(self):
        str = self.ui.timestrshow_2.text()
        # 解决方法1、
        # 这里因为inputMask设定的格式有年月日,所以转换时也要使用年月日才能返回正确的日期格式
        #dastr = QDate.fromString(str, "yyyy年MM月dd日")
        #self.ui.dateEdit.setDate(dastr)
        # 解决方法2、
        # 读取的文本包含年月日,首先需要去除年月日,调用了StrextractDigit函数取出数字,但是字符串
        # 字符串固定长度4+2+2 8位,进行切片即可得到年月日对应得整数,使用QDate()函数进行转换即可
        dastr1 = D.StrextractDigit(str) #字符串形如20200912
        year = int(dastr1[:4])
        month = int(dastr1[4:6])
        day = int(dastr1[6:])
        dastr2 = QDate(year,month,day)
        self.ui.dateEdit.setDate(dastr2)

    def on_setdatetimepushButton_clicked(self):
        str = self.ui.timestrshow_3.text()
        #print(str)
        tmdastr = QDateTime.fromString(str,"yyyy-MM-dd hh:mm:ss:zzz")
        self.ui.dateTimeEdit.setDateTime(tmdastr)

    '''读取日历组件的日期'''
    # 日历组件涉及到值的改变,不属于clicked信号,属于selectionChanged信号,C++原型中没有额外参数
    def on_calendarWidget_selectionChanged(self):
        date = self.ui.calendarWidget.selectedDate() # 日历选择的日期
        self.ui.selectdateshow.setText(date.toString("yyyy年M月d日"))

    #调整时间&日期组件时也能改变文本显示
    def on_timeEdit_timeChanged(self,qtime):
        #print(qtime) #QtCore.QTime(int,int,int)格式
        Qtime = QTime.toString(qtime,"hh:mm:ss")
        #必须是完整的,如果h:mm:ss可能导致界面组件选择为单数时如9:10:22会显示成90:10:22,"hh"就可以将9识别成09而不是90
        self.ui.timestrshow.setText(Qtime)
    def on_dateEdit_dateChanged(self,qdate):
        Qdate = QDate.toString(qdate,"yyyy-MM-dd")
        self.ui.timestrshow_2.setText(Qdate)

    def on_dateTimeEdit_dateTimeChanged(self,qtida):
        Qtida = QDateTime.toString(qtida,"yyyy-MM-dd hh:mm:ss:zzz")
        self.ui.timestrshow_3.setText(Qtida)

# 主程序
app = QApplication(sys.argv)
MyPushButton = QMyDateTime()
sys.exit(app.exec_())


