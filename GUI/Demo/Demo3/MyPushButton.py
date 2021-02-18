#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : MyPushButton.py
@Author : chenbei
@Date : 2020/10/4 20:51
'''
import sys
from PyQt5.QtWidgets import  QApplication,QWidget
from PyQt5.QtCore import  pyqtSlot ,Qt
from GUI.Demo.Demo3 import QPushButton_Use as QU
from PyQt5.QtGui import  QFont
class QMyPushButton(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.ui = QU.Ui_QPushButton_Use()
        self.ui.setupUi(self)
        self.show()
    # 参数性质 : C++原函数名(底层强制具有参数)
    # clicked信号 :不带参数
    # 使用默认clicked,不需要@pyqtSlot
    def on_btn_left_clicked(self):
        self.ui.editInput.setAlignment(Qt.AlignLeft) # 引用Qt的布局方法设置文本的对齐方式
    def on_btn_middle_clicked(self):
        self.ui.editInput.setAlignment(Qt.AlignCenter) # 居中
    def on_btn_right_clicked(self):
        self.ui.editInput.setAlignment(Qt.AlignRight) # 右对齐
    # 参数性质 : C++原函数名
    # clicked(bool) : 由于可以复选,所以需要checked参数,只要为checked为真就可以运行该程序
    @pyqtSlot(bool)
    def on_btn_bold_clicked(self,checked): # 粗体
        '''不能使用if checked ,虽然为真执行加粗,但是为假也就是不选中时仍然是加粗状态
        现在希望根据勾选状态实现加粗和不加粗,下边语句为假时设置不加粗是可以实现的
        斜体、下划线道理类似'''
        #font = self.ui.editInput.font()
        font = QFont() #也可以使用QFont类进行设置
        font.setBold(checked)
        font.setPointSize(18)
        self.ui.btn_bold.setAutoExclusive(False)
        self.ui.btn_bold.setCheckable(True)
        self.ui.editInput.setFont(font)
    @pyqtSlot(bool)
    def on_btn_italic_clicked(self,checked): # 斜体
        font = QFont() #也可以使用QFont类进行设置
        font.setItalic(checked)
        font.setPointSize(18)
        self.ui.btn_italic.setAutoExclusive(False)
        self.ui.btn_italic.setCheckable(True)
        self.ui.editInput.setFont(font)
    @pyqtSlot(bool)
    def on_btn_underline_clicked(self,checked):# 下划线
        font = QFont() #也可以使用QFont类进行设置
        font.setUnderline(checked)
        font.setPointSize(18)
        self.ui.btn_underline.setAutoExclusive(False)
        self.ui.btn_underline.setCheckable(True)
        self.ui.editInput.setFont(font)
    # 参数性质 : C++原函数名
    # clicked(bool) : 由于可以复选,所以需要checked参数,只要为checked为真就可以运行该程序
    @pyqtSlot(bool)
    def on_checkBox_clicked(self,checked): # 只读
        self.ui.editInput.setReadOnly(checked)
    @pyqtSlot(bool)
    def on_checkBox_2_clicked(self,checked): # 使能修改
        self.ui.editInput.setEnabled(checked)
    @pyqtSlot(bool)
    def on_checkBox_3_clicked(self,checked):
        self.ui.editInput.setClearButtonEnabled(checked)  # 显示清除按钮

# 主程序
app = QApplication(sys.argv)
MyPushButton = QMyPushButton()
sys.exit(app.exec_())



