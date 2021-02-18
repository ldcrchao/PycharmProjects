#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : MySimpleChart.py
@Author : chenbei
@Date : 2020/12/29 10:42
'''
import sys
from PyQt5.QtWidgets import  QApplication ,QMainWindow
from PyQt5.QtChart import QChartView , QChart , QLineSeries , QValueAxis
import pandas as pd
class QMySimpleChart(QMainWindow):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setWindowTitle('最简单的绘图程序')
        self.resize(580,420)
        chart = QChart()
        chart.setTitle('正弦和余弦曲线')
        chartview = QChartView(self)
        chartview.setChart(chart) # 图表添加到chartview
        self.setCentralWidget(chartview)

        y1 = QLineSeries()
        y2 = QLineSeries()
        y1.setName('正弦曲线')
        y2.setName('余弦曲线')
        chart.addSeries(y1)
        chart.addSeries(y2)

        X = pd.read_csv("C:/Users\chenbei\Desktop\钢\线圈数据-2k/240/csv/0.csv", encoding='gbk')
        X = X.values
        tem1 = X[:,0]
        tem2 = X[:,1]
        for i in range(len(tem1)) :
            h1 = tem1[i]
            y1.append(i/len(tem1),h1)
            h2 = tem2[i]
            y2.append(i/len(tem1),h2)

        '''
        t = 0
        intv = 0.1
        pointCount = 100
        for i in range(pointCount) :
            tem1 = math.cos(t)
            y1.append(t,tem1)
            tem2 = 1.5*math.sin(t+30)
            y2.append(t,tem2)
            t = t + intv
        '''

        axisX = QValueAxis()
        axisX.setRange(-0.1,1.5)
        axisX.setTitleText('时间/s')
        axisY = QValueAxis()
        axisY.setRange(-0.2,max(tem2)+0.2)
        axisY.setTitleText('幅值')

        chart.setAxisX(axisX,y1)
        chart.setAxisX(axisX,y2)
        chart.setAxisY(axisY,y1)
        chart.setAxisY(axisY,y2)
app = QApplication(sys.argv)
form = QMySimpleChart()
form.show()
sys.exit(app.exec_())
