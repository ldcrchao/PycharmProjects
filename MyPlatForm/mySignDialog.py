#%%
# -*- coding UTF-8 -*-
'''
@Project : MyProjects
@File : mySignDialog.py
@Author : chenbei
@Date : 2021/3/6 11:11
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

font_set = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=10.5)
import numpy as np
import pandas as pd
import  sys
from PyQt5.QtWidgets import  (QMenu,QDialog,QMainWindow,QAction ,QApplication ,QTableWidgetItem, QAbstractItemView,QLabel)
from PyQt5.QtCore import Qt,pyqtSlot,QTimer,QTime,QSize,pyqtSignal
from PyQt5.QtGui import  QIcon , QPainter,QFont,QPen,QColor,QBrush
from MyPlatform import signDialog
class mySignDialog(QDialog):
    signState = pyqtSignal(bool,bool)
    def __init__(self,parent=None):
        super().__init__(parent)
        self.ui = signDialog.Ui_signDialog()
        self.ui.setupUi(self)
        self.show()
    def checkUsernameIsTrue(self):
        if self.ui.username.text() == "admin" :
            return True
        else:
            return False
    def checkPasswordIsTrue(self):
        if self.ui.password.text() == "123456" :
            return True
        else:
            return False
    def __del__(self):
        self.close()
    @pyqtSlot()
    def on_sure_Button_clicked(self):
        #print("123")
        self.signState.emit(self.checkPasswordIsTrue(), self.checkUsernameIsTrue())
        #print(self.checkPasswordIsTrue(), self.checkUsernameIsTrue())
if __name__ == '__main__':

    app = QApplication(sys.argv)
    form = mySignDialog ()
    sys.exit(app.exec_())