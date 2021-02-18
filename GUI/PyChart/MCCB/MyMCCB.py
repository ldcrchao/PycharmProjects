#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : MyMCCB.py
@Author : chenbei
@Date : 2020/12/29 17:15
'''
from matplotlib.pylab import mpl
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Times New Roman'] # 设置字体风格,必须在前然后设置显示中文
mpl.rcParams['font.size'] = 10.5 # 图片字体大小
mpl.rcParams['font.sans-serif'] = ['SimHei','SongTi'] # 显示中文的命令
mpl.rcParams['axes.unicode_minus'] = False  #  显示负号的命令
plt.rcParams['image.interpolation'] = 'nearest' # 设置 interpolation style
plt.rcParams['savefig.dpi'] = 600 # 图片像素
plt.rcParams['figure.dpi'] = 400 # 分辨率
from matplotlib.font_manager import FontProperties
font_set = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=10.5) # matplotlib内无中文字节码，需要自行手动添加
from sklearn.preprocessing import LabelEncoder
import math
from time import time
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import sys
import numpy as np
from PyQt5.QtWidgets import  (QApplication ,QMainWindow , QTreeWidgetItem
                              ,QFileDialog ,QDockWidget )
from GUI.PyChart.MCCB import QMyMCCB
from PyQt5.QtCore import  pyqtSlot ,Qt ,QDir ,QFileInfo
from PyQt5.QtGui import  QIcon , QPixmap , QFont,QPainter
from PyQt5.QtChart import QChartView , QChart , QLineSeries , QValueAxis
import pandas as pd
class mymccb(QMainWindow):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.ui = QMyMCCB.Ui_QMyMCCB()
        self.ui.setupUi(self)
        self._chart = QChart()
        self._chart.setTitle('故障和正常波形')
        self.ui.graphicsView.setChart(self._chart) # 此条很重要，否则图表不显示！！！！
        self.ui.graphicsView.setRenderHint(QPainter.Antialiasing)
        self._chart.legend().setAlignment(Qt.AlignBottom)
        self.data = pd.read_csv("C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/FirstLevelPCA.csv", encoding='gbk')
        self.X , self.y = self.ReturnX_y()
        self.show()
    def ReturnStatic(self):
        data = pd.read_csv("C:/Users\chenbei\Documents\python数据\PYQT5_UI文件夹\绘图\MCCB\QMyMCCB/220.csv")
        X = data.values
        X_kase = X[:,1]
        x_mean = np.mean(X_kase)
        x_std = np.std(X_kase)
        x_rms = math.sqrt(pow(x_mean, 2) + pow(x_std, 2))
        x_skew = np.mean((X_kase - x_mean) ** 3) / pow(x_std, 3)  # 偏度
        x_kurt = np.mean((X_kase - x_mean) ** 4) / pow(np.var(X_kase), 2)  # 峭度
        List = [round(x_mean,3),round(x_rms,3),round(x_skew,3),round(x_kurt,3)]
        return List
    def ReturnX_y(self):
        X_dataframe = self.data.iloc[:, 0:-1]  # 分出数据和标签 此时是DataFrame格式
        y_dataframe = self.data.iloc[:, -1]
        X = X_dataframe.values  # ndarray格式 样本数×维数
        y_category = y_dataframe.values  # ndarray格式
        Label = LabelEncoder()  # 初始化1个独热编码类
        y = Label.fit_transform(y_category)  # 自动生成标签
        return X,y
    def SVMReturn(self):
        start = time()
        clf = svm.SVC(kernel='linear', C=1, probability=True)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, train_size=0.9)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        pp = 0
        for j in range(len(y_pred)):
            if y_pred[j] == y_test[j]:
                pp = pp + 1
        pp = pp / len(y_pred)  # 某一次的预测准确率
        tem = list(np.random.rand(len(y_pred)))
        loc = tem.index(max(tem))
        loc1 = int(loc / 6)
        test_X = X_test[loc,:] # 测试样本的第loc个样本
        test_X = test_X.reshape(1,2)
        test_pre = clf.predict(test_X)
        if test_pre == 0 :
            label = 'C0'
        elif test_pre == 1 :
            label = 'C1'
        elif test_pre == 2 :
            label = 'M4'
        elif test_pre == 3 :
            label = 'M9'
        elif test_pre == 4 :
            label = 'H3'
        elif test_pre == 5 :
            label = 'H6'
        elif test_pre == 6 :
            label = 'C3'
        elif test_pre  == 7 :
            label = 'M7'
        if loc1 == 0:
            label1 = 'C0'
        elif loc1 == 1:
            label1 = 'C1'
        elif loc1 == 2:
            label1 = 'M4'
        elif loc1 == 3:
            label1 = 'M9'
        elif loc1 == 4:
            label1 = 'H3'
        elif loc1 == 5:
            label1 = 'H6'
        elif loc1 == 6:
            label1 = 'C3'
        elif loc1 == 7:
            label1 = 'M7'
        end = time()
        return str(pp),label,str((pp/2+(1-pp))),label1 , str(round((end-start)*100,5))

    @pyqtSlot()
    def on_LoadData_clicked(self) :
        #self.setCentralWidget(chartview)
        y1 = QLineSeries()
        y2 = QLineSeries()
        y2.setName('故障波形')
        y1.setName('正常波形')
        self._chart.addSeries(y1)
        self._chart.addSeries(y2)
        data = pd.read_csv("C:/Users\chenbei\Documents\python数据\PYQT5_UI文件夹\绘图\MCCB\QMyMCCB/220.csv")
        X = data.values
        X_Normal = X[:,0]
        X_kase = X[:,1]
        for i in range(len(X_Normal)) :
            h1 = X_Normal[i]
            y1.append(i / len(X_Normal), h1)
            h2 = X_kase[i]
            y2.append(i / len(X_kase), h2)
        axisX = QValueAxis()
        axisX.setRange(0.0,1.0)
        axisX.setTitleText('时间/s')
        axisY = QValueAxis()
        axisY.setRange(-0.2,max(max(X_kase),max(X_Normal))+0.2)
        axisY.setTitleText('幅值/A')
        self._chart.setAxisX(axisX,y1)
        self._chart.setAxisX(axisX,y2)
        self._chart.setAxisY(axisY,y1)
        self._chart.setAxisY(axisY,y2)

    @pyqtSlot()
    def on_StartTest_clicked(self):
        statics = self.ReturnStatic()
        self.ui.piandu.setText(str(statics[2]))
        self.ui.qiaodu.setText(str(statics[3]))
        self.ui.AVA.setText(str(statics[0]))
        self.ui.lineEdit_6.setText(str(statics[1]))
        Checked = self.ui.SVM.isChecked()
        if Checked :
            prob1,label1,prob2,label2 ,Time= self.SVMReturn()
            self.ui.code1.setText(label1)
            self.ui.code2.setText(label2)
            self.ui.prob1.setText(prob1)
            self.ui.prob2.setText(prob2)
            self.ui.consumetime.setText(Time)



app = QApplication(sys.argv)
form = mymccb()
sys.exit(app.exec_())
