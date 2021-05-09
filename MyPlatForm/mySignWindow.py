#%%
# -*- coding UTF-8 -*-
'''
@Project : MyProjects
@File : mySignWindow.py
@Author : chenbei
@Date : 2021/3/5 20:51
'''
import matplotlib.pyplot as plt
from matplotlib.pylab import mpl

plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 设置字体风格,必须在前然后设置显示中文
mpl.rcParams['font.size'] = 10.5  # 图片字体大小
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文的命令
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号的命令
mpl.rcParams['agg.path.chunksize'] = 10000
plt.rcParams['figure.figsize'] = (7.8, 3.8)  # 设置figure_size尺寸
plt.rcParams['savefig.dpi'] = 600  # 图片像素
plt.rcParams['figure.dpi'] = 600  # 分辨率
from matplotlib.font_manager import FontProperties
import  sys
font_set = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=10.5)
import numpy as np
import pandas as pd
import os
from PyQt5.QtWidgets import  (QLineEdit,QMenu,QDialog,QMainWindow,QAction ,QApplication ,QTableWidgetItem, QAbstractItemView,QLabel)
from PyQt5.QtCore import Qt,pyqtSlot,QTimer,QTime,QSize,pyqtSignal
from PyQt5.QtGui import  QIcon , QPainter,QFont,QPen,QColor,QBrush,QPalette,QPixmap,QMovie
from MyPlatform import signWindow
class mySignWindow(QMainWindow):
    signState = pyqtSignal(bool,bool)
    def __init__(self,parent=None):
        super().__init__(parent)
        self.ui = signWindow.Ui_signIn_MainWindow()
        self.ui.setupUi(self)
        self.ui.password.setEchoMode(QLineEdit.Password)
        self.ui.username.setEchoMode(QLineEdit.Password)
        # gif动画
        # path = os.getcwd() + "/MyPlatform/Login.gif"
        # movie = QMovie(path)
        # # 设置cacheMode为CacheAll时表示gif无限循环，注意此时loopCount()返回-1
        # movie.setCacheMode(QMovie.CacheAll)
        # # 播放速度
        # movie.setSpeed(100)


        # print(self.width(),self.height())
        # self.ui.label.setMinimumWidth(2000)
        # self.ui.label.setMaximumWidth(2000)
        # self.ui.label.setMinimumHeight(1000)
        # self.ui.label.setMaximumHeight(1000)
        # self.ui.label.setFixedWidth(2000)
        # self.ui.label.setFixedHeight(1000)
        #self.ui.label.setMovie(movie)

        # 开始播放，对应的是movie.start()
        #movie.start()

        # 设置背景图片的操作
        palette = QPalette()
        path1 = os.getcwd() + "/MyPlatform/lover.jpg"
        pix = QPixmap(path1)
        pix.scaled(self.width(),self.height())
        palette.setBrush(QPalette.Background, QBrush(pix))
        self.setPalette(palette)
        self.show()
    def checkUsernameIsTrue(self):
        if self.ui.username.text() == "null" :
            return True
        else:
            return False
    def checkPasswordIsTrue(self):
        if self.ui.password.text() == "1998" :
            return True
        else:
            return False
    def __del__(self):
        self.close()
    @pyqtSlot()
    def on_sure_Button_clicked(self):
        self.signState.emit(self.checkPasswordIsTrue(), self.checkUsernameIsTrue())
if __name__ == '__main__':

    app = QApplication(sys.argv)
    form = mySignWindow ()
    sys.exit(app.exec_())