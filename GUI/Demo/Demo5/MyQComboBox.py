#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : MyQComboBox.py
@Author : chenbei
@Date : 2020/10/8 19:27
'''
import sys
from PyQt5.QtWidgets import  QApplication ,QWidget
from PyQt5.QtCore import  pyqtSlot,Qt
from PyQt5.QtGui import  QIcon
from GUI.Demo.Demo5 import QComboBox

class MyQComboBox(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.ui = QComboBox.Ui_QComboBox()
        self.ui.setupUi(self)
        #self.ui.Enabled.setEnabled(True)
        self.show()

    def on_initiallist_clicked(self):
        '''程序内部先定义初始化的列表'''
        jpg = QIcon(":/images/images/aim.ico")
        self.ui.comboBox1.clear() # 点击初始化列表按钮时先清楚以前的状态
        citys = ["北京市","上海市","厦门市","重庆市","南京市","成都市","济南市","大连市","长沙市","武汉市","南昌市"]
        for i in range(len(citys)):
            self.ui.comboBox1.addItem(jpg,citys[i]) # 循环添加项目及其图标
        for idx , element in enumerate(citys): # 使用枚举类型
            self.ui.comboBox1.addItem(jpg,citys[idx]) # 第一种参数支持传递图标和文字

    def on_clearlist_clicked(self):
        self.ui.comboBox1.clear()

    @pyqtSlot(bool)
    def on_Enabled_clicked(self,checked):
        '''当复选框勾选时，下拉列表选择某一项后变成LineEdit可以输入内容，否则禁用'''
        self.ui.comboBox1.setEnabled(checked)

    @pyqtSlot(str)
    def on_comboBox1_currentIndexChanged(self,text):
        '''下拉列表框选择项时下拉项索引改变和值传递的信号为currentIndexChanged(int/str)
        属于overload信号,int传递索引,str传递对应文字'''
        self.ui.lineEdit.setText(text) #把下拉列表的某一项文字输入到文本上
        self.ui.lineEdit.setAlignment(Qt.AlignVCenter)
        self.ui.lineEdit.setAlignment(Qt.AlignHCenter)

    '''以上的下拉列表框不带用户数据，只有图标和文字'''
    def on_initial_city_num_clicked(self):
        jpg = QIcon(":/images/images/aim.ico")
        self.ui.comboBox2.clear()
        citys = {"北京市":10,"上海市":21,"厦门市":592,"重庆市":23,"南京市":25,"成都市":28,"济南市":521,
                "大连市":411,"长沙市":731,"武汉市":27,"南昌市":791}
        for k in citys:
            self.ui.comboBox2.addItem(jpg,k,citys[k]) # 第二种参数支持传递图标、文字和用户数据
            # 这里k是keys，citys[k]是values

    @pyqtSlot(str)
    def on_comboBox2_currentIndexChanged(self,text):
        self.ui.lineEdit.setText(text)
        zone = self.ui.comboBox2.currentData()  # 读取的是字典的values,即用户数据
        if (zone != None) :
            self.ui.lineEdit.setText(text+":区号= %d" % zone)
            self.ui.lineEdit.setAlignment(Qt.AlignVCenter)
            self.ui.lineEdit.setAlignment(Qt.AlignHCenter)

app = QApplication(sys.argv)
form = MyQComboBox()
sys.exit(app.exec_())