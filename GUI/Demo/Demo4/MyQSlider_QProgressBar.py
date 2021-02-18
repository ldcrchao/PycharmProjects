#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : MyQSlider_QProgressBar.py
@Author : chenbei
@Date : 2020/10/5 9:56
'''
import sys
from PyQt5.QtWidgets import  QApplication,QWidget
from PyQt5.QtCore import  pyqtSlot
from GUI.Demo.Demo4 import QSilder_QProgressBar as QSP
class QMySilder_progressbar(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.ui = QSP.Ui_QSilder_QProgressBar()
        self.ui.setupUi(self)
        self.show()

        self.ui.horizontalSlider.setMaximum(200)
        self.ui.horizontalScrollBar.setMaximum(200)
        self.ui.progressBar.setMaximum(200)

        self.ui.horizontalSlider.valueChanged.connect(self.do_valueChanged) #valueChanged需要连接,clicked不需要
        self.ui.horizontalScrollBar.valueChanged.connect(self.do_valueChanged)
        # 以下4条可以省略
        self.ui.percent.clicked.connect(self.on_percent_clicked)
        self.ui.recentvalue.clicked.connect(self.on_recentvalue_clicked)
        self.ui.textVisible.clicked.connect(self.on_textVisible_clicked)
        self.ui.InvertedAppearance.clicked.connect(self.on_InvertedAppearance_clicked)
    # valueChanged(int) -> QSlider和QScrollBar的内建信号
    def do_valueChanged(self,value):
        self.ui.progressBar.setValue(value)
    # clicked() -> radiobutton 类
    def on_percent_clicked(self):
        self.ui.progressBar.setFormat("%p%") # 设置百分比模式,"%v"显示当前值,"%m"显示总步数
    def on_recentvalue_clicked(self):
        self.ui.progressBar.setFormat("%v")
    # clicked(bool) -> checkbox 类
    @pyqtSlot(bool)
    def on_InvertedAppearance_clicked(self,checked):
        self.ui.progressBar.setInvertedAppearance(checked)
    @pyqtSlot(bool)
    def on_textVisible_clicked(self,checked):
        self.ui.progressBar.setTextVisible(checked)
# 主程序
app = QApplication(sys.argv)
MyPushButton = QMySilder_progressbar()
sys.exit(app.exec_())
