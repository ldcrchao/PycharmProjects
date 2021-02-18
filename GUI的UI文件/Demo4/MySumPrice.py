#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : MySumPrice.py
@Author : chenbei
@Date : 2020/10/4 11:05
'''
import sys
from GUI.Demo.Demo2 import SumPrice as SP
from PyQt5.QtWidgets import  QApplication ,QWidget
from PyQt5.QtCore import  pyqtSlot
class QMySumPrice(QWidget):
    #signals = pyqtSignal(float) ,自定义信号必须使用QObject类
    def __init__(self,parent=None,sumprice=float(2500)):
        super().__init__(parent)
        self.ui = SP.Ui_SumPrice()
        self.ui.setupUi(self) # 构建窗体
        self.do_changesumprice(sumprice) # 内部传递,不必定义类似于human的类,进行复杂的传递,直接定义方法改变即可
        #self.emitvalue(sumprice)
    '''
    def emitvalue(self,sumprice):
        print(sumprice)
        self.signals.emit(sumprice)
    def changevalue(self):
        num = self.sumprice
        print(num)
        self.ui.autocalpricedoubleSpinBox.setValue(num)
    '''
    def on_CalculateButton_clicked(self):
        # 点击计算总价按钮后执行
        num = int(self.ui.num.text()) # 读取数量的文本
        price = float(self.ui.price.text()) # 价格
        self.ui.sumprice.setText("%.2f" %(num*price))

    @pyqtSlot(int)
    def on_numspinBox_valueChanged(self,count):
        '''之所以可以使用count参数,是因为在C++函数原型中此函数就有输入参数,已经将下拉值的变化转换成输入参数'''
        price = self.ui.pricedoubleSpinBox.value() # 读取价格的值
        self.ui.autocalpricedoubleSpinBox.setValue(count*price)

    @pyqtSlot(float)
    def on_pricedoubleSpinBox_valueChanged(self,price):
        count = self.ui.numspinBox.value() # 读取数量的值
        self.ui.autocalpricedoubleSpinBox.setValue(count*price)

    def do_changesumprice(self,sumprice):
        self.ui.autocalpricedoubleSpinBox.setValue(sumprice)

# 主程序
app = QApplication(sys.argv)
form = QMySumPrice(sumprice=4000)
#form.signals.connect(form.changevalue)
form.show()
app.exit(app.exec_())
