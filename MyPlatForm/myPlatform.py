# %%
# -*- coding UTF-8 -*-
'''
@Project : MyProjects
@File : myPlatform.py
@Author : chenbei
@Date : 2021/3/1 21:32
'''
# 特征提取模块
from Feature_extraction.Features import  emd_entropy , svd_entropy, permutation_entropy
from pyentrp import entropy as ent
from vectorizedsampleentropy import vectapen
# 特征预测模块
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import validation_curve  # 模型评估 : 不同内核系数gamma
from sklearn.model_selection import learning_curve  # 模型评估 : 学习率曲线
from sklearn.model_selection import cross_val_score  # 模型评估 : 不同正则化系数C
from sklearn.model_selection import LeavePOut  # 留P法, K折交叉验证中K=n-p的情况,即剩下p个测试集
from sklearn.model_selection import LeaveOneOut  # 留1法, K折交叉验证中K=n(样本数)的情况
from sklearn.model_selection import KFold  # K折交叉验证
from sklearn.model_selection import train_test_split  # 留出法的分割方式
from sklearn.model_selection import ShuffleSplit# ShuffleSplit方法，可以随机的把数据打乱，然后分为训练集和测试集
from sklearn.kernel_approximation import (RBFSampler, Nystroem)  # 内核近似
from sklearn import metrics
from sklearn import pipeline
from sklearn.preprocessing import LabelEncoder

# Qt相关模块
from PyQt5.QtWidgets import (QMainWindow,QAction,QApplication,QMessageBox,
                             QTableWidgetItem,QAbstractItemView,QLabel)
from PyQt5.QtChart import QChart, QLineSeries, QValueAxis
from PyQt5.QtGui import QIcon, QPainter, QFont, QPen, QColor, QBrush, QPalette, QPixmap
from PyQt5.QtCore import Qt, pyqtSlot, QTime, QSize
from MyPlatform.mySignDialog import mySignDialog
from MyPlatform.mySignWindow import mySignWindow
from MyPlatform import faultPlatform

import os
import sys
from enum import Enum
import random
import numpy as np
import pandas as pd
import math
import time
import seaborn as sns
from scipy.fftpack import fft
# 绘制格式
from matplotlib.font_manager import FontProperties
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
font_set = FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=10.5)
class cellType(Enum):
    # 定义参数单元格类型为1  参数大小单元格类型为2
    # 用于循环添加项时指定添加的项类型 因为同一行不同列的数据类型可能不同
    parameter_name_type = 1
    parameter_value_type = 2

class columnIndex(Enum):
    # 用于循环每行赋值时指定单元格的列号 参数第一列 值第二列 必须按顺序排列
    parameter_name_num = 0
    parameter_value_num = 1

class myPlatform(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = faultPlatform.Ui_faultPlatform()
        self.ui.setupUi(self)

        self.init_comboBox_text_icon()  # 初始化comboBox
        self.init_waveform()  # 初始化波形显示
        self.cancel_training_flag = False  # 当为真时训练会退出程序
        # 只要点击了加载波形就设置为真 用于后续作为菜单栏能否初始化的判据
        # 界面只有加载波形、开始训练可以改变界面状态 所以这两个用作判据 取消训练不作为判据
        self.have_trained_model = False  # 从未点击过开始训练 为False 按钮clicked时设为True
        self.have_loaded_waveform = False  # 从未加载过波形 为False 按钮clicked时设为True
        self.init_menuAction_buttonClicked()  # 初始化菜单栏和按钮的联系 菜单栏的triggered连接按钮的btn
        self.init_other_properties()  # 初始化其他相关属性

        self._init_tableWidget = False  # 波形时域特征表格未初始化
        if not self._init_tableWidget:  # 未初始化时采取初始化
            # 初始化成功后self._init_tableWidget会更新为True 不会再进行初始化 防止comboBox选择时重复初始化
            self.init_table_time_features()  # 初始化波形参数

        self._init_current_tableWidget = False  # 电流故障表格未初始化
        if not self._init_current_tableWidget:
            self.init_current_tableWidget()  # 初始化电流预测概率表格

        self._init_vibration_tableWidget = False  # 振动故障表格未初始化
        if not self._init_vibration_tableWidget:
            self.init_vibration_tableWidget()  # 初始化振动预测概率表格

        # 在这里初始化动作才能被按钮使用 局部变量初始化会存在动态动作无法获取的问题
        self.init_action = QAction(
            QIcon(":/images/refresh.ico"), "初始化窗口", self)
        self.init_widget_menu()  # 初始化动态的菜单

        self.close()

        # 登录窗口
        # 1、使用继承QMainWindow继承的登陆界面 不能设置模态特性
        self.mySignWindow = mySignWindow(self)
        self.mySignWindow.setAttribute(Qt.WA_DeleteOnClose)  # 主窗口关闭时自动删除
        self.mySignWindow.signState.connect(self.signWindowIsSuccess)

        # 2、改用继承QDialog类的登陆界面
        # self.mySignDialog = mySignDialog(self)
        # self.mySignDialog.setAttribute(Qt.WA_DeleteOnClose)  # 主窗口关闭时自动删除
        # self.mySignDialog.signState.connect(self.signDialogIsSuccess)

    '''1、界面组件类函数'''
    # 1、设置combox的图标和文字
    def init_comboBox_text_icon(self):
        self.ui.comboBox.clear()
        text = ["线圈欠压160V", "线圈欠压180V", "线圈欠压200V", "线圈正常220V",
                "线圈顶杆卡涩", "线圈轻微老化", "线圈严重老化", "线圈过压240V",
                "正常合闸","正常分闸",  "虚假合闸", "分闸不彻底",
                "储能弹簧卡涩","操作机构卡涩","合闸螺栓松动", "分闸螺栓松动"]
        for i in range(16):
            path = ":/images/" + str(i + 1) + ".jpg"
            icon = QIcon(path)
            self.ui.comboBox.setIconSize(QSize(96, 96))
            self.ui.comboBox.addItem(icon, text[i])
        self.ui.comboBox.setCurrentText("线圈正常220V")

    # 2、初始化正常波形的函数 初始化时只显示1个波形 要求init=True 由于显示一个即可 故不再显示振动
    def init_waveform(self):
        if self.set_waveform_parameters(
                type="电流",
                title="正常分合闸电流信号(220V)",
                init=True):
            print("波形初始化成功...")
            label = QLabel(self)  # 状态栏必须动态添加
            label.setMinimumWidth(250)
            label.setText("分合闸电流的正常波形初始化成功...")
            self.ui.statusbar.addWidget(label)  # 将状态栏显示到窗口上
            font = QFont()
            font.setFamily("TimeNewRoman")
            font.setBold(True)
            font.setPointSize(15)
            self.ui.statusbar.setFont(font)
        else:
            print("初始化失败")

    # 3、设置波形参数的函数,每次绘图都可以进行调用
    def set_waveform_parameters(self, type, title, option=None, init=False):
        '''
        :param type: 选择电流还是振动波形展示 函数只在初始化调用一次和显示表格参数调用一次
        初始化波形默认使用电流波形,故不再提供振动信号的代码
        :param title: 图表标题
        :param option: 选择故障类型(不选择正常波形)
        :param init: 如果初始化参数为True 表示只展示正常波形, 否则正常和故障波形一起展示
        :return: True
        '''
        y = QLineSeries()  # 电流信号或振动信号的正常波形
        y.setName('正常波形')

        # 1、设置线型、颜色、宽度
        open = QPen()
        open.setColor(Qt.red)
        open.setStyle(Qt.SolidLine)
        open.setWidth(2)
        y.setPen(open)
        chart = QChart()
        # 2、设置图表主题
        # 0: light主题(默认) 1 :蔚蓝主题 2:暗黑主题 3:沙褐色主题 4 :自然色彩系统（NCS）蓝色主题
        # 5 : 高对比度主题 6:冰蓝色的主题 7:Qt 主题
        chart.setTheme(QChart.ChartTheme(1))
        # 3、设置标题加粗、黑色、大小、颜色、字体
        chart.setTitle(title)
        font = QFont()
        font.setBold(True)
        font.setFamily("TimeNewRoman")
        font.setPointSize(18)
        chart.setTitleFont(font)
        chart.setTitleBrush(QBrush(QColor(Qt.black)))
        # 4、设置图例位置、大小、文字颜色
        chart.legend().setAlignment(Qt.AlignBottom)
        chart.addSeries(y)
        self.ui.graphicsView.setChart(chart)
        chart.legend().setLabelColor(QColor(Qt.black))
        font.setPointSize(14)
        chart.legend().setFont(font)
        self.ui.graphicsView.setRenderHint(QPainter.Antialiasing)
        # 5、设置图表动画效果
        chart.setAnimationOptions(QChart.AllAnimations)

        if init:  # 如果只需要展示正常波形 用于界面初始化时显示
            if type == "电流":
                current_normal = self.spilit_current_waveform_data(
                    "220").values  # 这里返回的是series
                for i in range(len(current_normal)):
                    point = current_normal[i]
                    # 第一项参数决定的是以多大步长添加点 如第1个点(0,point)第2个点(0.05/1100)最后一个点(0.05,point)
                    y.append(i * 0.05 / len(current_normal), point)
                # 6、设置坐标轴范围、主分隔数、网格线(是否显示、颜色)、标签格式(字体、大小、颜色、精度)
                axisX = QValueAxis()
                axisX.setRange(0.0, 0.05)
                axisX.setTitleText('时间/s')
                axisX.setTickCount(10)  # 主分隔数
                axisX.setGridLineVisible(False)
                axisX.setGridLineColor(QColor(Qt.gray))
                axisX.setLabelFormat("%.2f")
                font = QFont()
                font.setFamily("TimeNewRoman")
                font.setPointSize(12)
                axisX.setLabelsFont(font)
                axisX.setLabelsColor(QColor(Qt.black))
                axisY = QValueAxis()
                axisY.setRange(-0.2, max(current_normal) + 0.2)
                axisY.setTitleText('幅值/A')
                axisY.setTickCount(5)
                axisY.setGridLineVisible(True)
                axisY.setLabelFormat("%.2f")
                axisY.setLabelsFont(font)
                axisY.setLabelsColor(QColor(Qt.black))
                chart.setAxisX(axisX, y)
                chart.setAxisY(axisY, y)
                return True
            elif type == "振动":
                 pass # 界面初始化波形不需要振动 所以这里就不写了 也不会调用
            else:
                print("类型type只能输入<电流>或<振动>！")
                return False
        else:  # 故障波形和正常波形一起显示
            if type == "电流":
                # 设置正常波形
                current_normal = self.spilit_current_waveform_data(
                    "220")  # 这里返回的是series
                for i in range(len(current_normal)):
                    point = current_normal[i]
                    # 第一项参数决定的是以多大步长添加点 如第1个点(0,point)第2个点(0.05/1100)最后一个点(0.05,point)
                    y.append(i *0.05 / len(current_normal), point)
                open = QPen()
                open.setColor(Qt.red)
                open.setStyle(Qt.SolidLine)
                open.setWidth(2)
                y.setPen(open)
                # 设置故障波形
                y0 = QLineSeries()
                chart.addSeries(y0)
                y0.setName('故障波形')
                open.setColor(Qt.green)
                y0.setPen(open)
                current_fault = self.spilit_current_waveform_data(option)
                # 第一项参数决定的是以多大步长添加点 如第1个点(0,point)第2个点(0.05/1100)最后一个点(0.05,point)
                for i in range(len(current_fault)):
                    point = current_fault[i]
                    # 也可以使用 i / 20000      0.05/1000 = 1/20000
                    y0.append(i *0.05 / len(current_fault), point)
                font = QFont()
                font.setFamily("TimeNewRoman")
                font.setPointSize(12)

                axisX = QValueAxis()
                axisX.setRange(0.0, 0.05)
                axisX.setTitleText('时间/s')
                axisX.setTickCount(10)
                axisX.setGridLineVisible(False)
                axisX.setGridLineColor(QColor(Qt.gray))
                axisX.setLabelFormat("%.2f")
                axisX.setLabelsFont(font)
                axisX.setLabelsColor(QColor(Qt.black))
                axisY = QValueAxis()
                axisY.setRange(-0.2, max(max(current_normal),
                                         max(current_fault)) + 0.2)
                axisY.setTitleText('幅值/A')
                axisY.setTickCount(5)
                axisY.setGridLineVisible(True)
                axisY.setLabelFormat("%.2f")
                axisY.setLabelsFont(font)
                axisY.setLabelsColor(QColor(Qt.black))
                chart.setAxisX(axisX, y)
                chart.setAxisY(axisY, y)
                chart.setAxisX(axisX, y0)
                chart.setAxisY(axisY, y0)
                return True
            elif type == "振动":
                # "正常合闸","正常分闸",  "虚假合闸", "分闸不彻底",
                # "储能弹簧卡涩","操作机构卡涩","合闸螺栓松动", "分闸螺栓松动"
                # 对振动波形不再同时显示正常波形和故障波形,只单独显示
                vibration_waveform = self.spilit_vibration_waveform_data(
                    option).values  # 这里返回的是series
                N =  100000 # 采样率 用于转换时间 使用波形长度/采样率 = 时间 4500/N=0.045s 10000/N=0.1s
                for i in range(len(vibration_waveform)):
                    point = vibration_waveform[i]
                    # 第一项参数决定的是以多大步长添加点 如第1个点(0,point)第2个点(1/10000)最后一个点(0.045,point)
                    y.append(i / N, point)
                # 6、设置坐标轴范围、主分隔数、网格线(是否显示、颜色)、标签格式(字体、大小、颜色、精度)
                open = QPen()
                open.setColor(Qt.red)
                open.setStyle(Qt.SolidLine)
                open.setWidth(2)
                y.setPen(open)
                axisX = QValueAxis()
                axisX.setRange(0.0, len(vibration_waveform)/N)
                axisX.setTitleText('时间/s')
                axisX.setTickCount(10)  # 主分隔数
                axisX.setGridLineVisible(False)
                axisX.setGridLineColor(QColor(Qt.gray))
                axisX.setLabelFormat("%.2f")
                font = QFont()
                font.setFamily("TimeNewRoman")
                font.setPointSize(12)
                axisX.setLabelsFont(font)
                axisX.setLabelsColor(QColor(Qt.black))
                axisY = QValueAxis()
                axisY.setRange(-0.2, max(vibration_waveform) + 0.2)
                axisY.setTitleText('幅值/V')
                axisY.setTickCount(5)
                axisY.setGridLineVisible(True)
                axisY.setLabelFormat("%.2f")
                axisY.setLabelsFont(font)
                axisY.setLabelsColor(QColor(Qt.black))
                chart.setAxisX(axisX, y)
                chart.setAxisY(axisY, y)
                return True
            else:
                print("类型type只能输入<电流>或<振动>！")
                return False

    # 4、combox的槽函数 根据当前项的变化 显示到状态栏
    @pyqtSlot(str)
    def on_comboBox_currentIndexChanged(self, comboBox_item_text):
        # self.ui.statusbar.clearMessage()
        # 1、根据下拉波形的选择改变状态栏信息
        self.ui.statusbar.update()
        self.ui.statusbar.showMessage("当前为" + comboBox_item_text + "的波形")
        font = QFont()
        font.setFamily("TimeNewRoman")
        font.setBold(True)
        font.setPointSize(15)
        self.ui.statusbar.setFont(font)
        # 2、根据下拉波形的选择改变波形特征参数的值
        # 项的选择整型编号变化--->字符串编号--->计算波形参数
        comboBox_item_int = self.ui.comboBox.currentIndex()  # 依然找到当前选择的编号
        if self.convert_faultNumber_intToString(comboBox_item_int):
            # 转换整型编号--->字符编号
            comboBox_item_str = self.convert_faultNumber_intToString(
                comboBox_item_int)
            # 得到代号后继续判断是振动还是电流
            if comboBox_item_int < 8:
                # 1、 编号在0~7为电流信号
                #  计算电流信号指定option的波形特征参数
                if self.set_table_values("电流", comboBox_item_str):  # 先判断是否计算成功
                    print("计算当前电流波形的时域特征成功...")
                    features = self.set_table_values("电流", comboBox_item_str)

                    # 将这些参数传递给项第2列
                    for index, value in enumerate(features):
                        # index = 0~15 也作为索引项的行号 列号不变 然后将value也就是波形特征数据直接设置为值
                        # print(index,value)

                        self.ui.tableWidget.item(
                            index, columnIndex.parameter_value_num.value).setText(
                            str(value))
                        # print(self.ui.tableWidget.item(index,columnIndex.parameter_value_num.value).text())
                        # 同时更新用户数据
                        self.ui.tableWidget.item(
                            index, columnIndex.parameter_value_num.value).setData(
                            Qt.UserRole, str(value))
                        #print(self.ui.tableWidget.item(index, columnIndex.parameter_value_num.value).data(Qt.UserRole))
                        self._init_tableWidget = True  # 防止选取不同波形时重复初始化
                else:
                    print("计算当前电流波形的时域特征失败...")
            else:
                # 2、编号8~15为振动信号 功能后续添加
                if self.set_table_values("振动", comboBox_item_str):  # 先判断是否计算成功
                   print("计算当前振动波形的时域特征成功...")
                   features = self.set_table_values("振动", comboBox_item_str)
                   # 将这些参数传递给项第2列
                   for index, value in enumerate(features):
                       self.ui.tableWidget.item(
                           index, columnIndex.parameter_value_num.value).setText(
                           str(value))
                       # 同时更新用户数据
                       self.ui.tableWidget.item(
                           index, columnIndex.parameter_value_num.value).setData(
                           Qt.UserRole, str(value))
                       self._init_tableWidget = True  # 防止选取不同波形时重复初始化
                else:
                    print("计算当前振动波形的时域特征失败...")

        else:  # 转换整型失败时候返回
            self.ui.statusbar.clearMessage()
            self.ui.statusbar.showMessage("波形时域特征参数加载失败...")

    # 5、加载波形QPushButton的槽函数
    @pyqtSlot()
    def on_pushButton_clicked(self):
        # 只要点击了加载波形就设置为真 用于后续作为菜单栏能否初始化的判据
        self.have_loaded_waveform = True
        # 只要点击过按钮就可以将菜单初始化动作使能
        if self.have_loaded_waveform or self.have_trained_model:
            self.init_action.setEnabled(True)  # 禁用动作
            self.init_action.setChecked(True)
        else:
            self.init_action.setEnabled(False)  # 禁用动作
            self.init_action.setChecked(False)
        # 检测到combox的变化加载对应的波形 根据当前编号检测
        comboBox_item_int = self.ui.comboBox.currentIndex()
        # print(comboBox_item_int)
        # 调用函数将故障整型编号--->字符串代号 因为加载波形需要代号
        if self.convert_faultNumber_intToString(comboBox_item_int):
            comboBox_item_str = self.convert_faultNumber_intToString(
                comboBox_item_int)
            # print(comboBox_item_str)
            # 得到代号后继续判断是振动还是电流
            if comboBox_item_int < 8:
                # 1、 编号在0~7为电流信号
                # 得到转换后代号对应的电流数据
                if self.set_waveform_parameters(
                    type="电流",
                    title=self.ui.comboBox.currentText() +
                    "电流信号",
                    option=comboBox_item_str,
                        init=False):
                    self.ui.statusbar.clearMessage()
                    self.ui.statusbar.showMessage("波形加载成功")

            else:
                # 2、编号8~15为振动信号 功能后续添加
                if self.set_waveform_parameters(
                    type="振动",
                    title=self.ui.comboBox.currentText()+"振动信号",
                    option = comboBox_item_str,
                    init = False) :
                    self.ui.statusbar.clearMessage()
                    self.ui.statusbar.showMessage("波形加载成功")
        else:  # 转换整型失败时候返回
            self.ui.statusbar.clearMessage()
            self.ui.statusbar.showMessage("波形加载失败...")

    # 6、初始化时域特征表格tableWidget 用电流的初始化
    def init_table_time_features(self):
        # 1、设置表头、列数 、是否可编辑、交替行背景、自动调整行高列宽
        self.ui.tableWidget.setAlternatingRowColors(True)  # 交替行背景
        self.ui.tableWidget.resizeRowsToContents()  # 行高
        # self.ui.tableWidget.resizeColumnsToContents() # 列宽
        self.ui.tableWidget.setColumnWidth(
            columnIndex.parameter_name_num.value, 508)
        self.ui.tableWidget.setColumnWidth(
            columnIndex.parameter_value_num.value, 200)
        headerText = ["参数", "大小"]
        self.ui.tableWidget.setColumnCount(len(headerText))  # 表格列数
        self.ui.tableWidget.setEditTriggers(
            QAbstractItemView.NoEditTriggers)  # 不可编辑
        for i in range(len(headerText)):
            #  设置表头
            headerItem = QTableWidgetItem(headerText[i])
            # 设置表头格式
            font = headerItem.font()
            font.setBold(True)
            font.setPointSize(15)
            font.setFamily("楷体")
            headerItem.setFont(font)
            # 设置对齐方式
            headerItem.setTextAlignment(Qt.AlignVCenter)
            headerItem.setTextAlignment(Qt.AlignHCenter)
            # 文字颜色
            headerItem.setForeground(QBrush(Qt.black))
            # headerItem.setBackground(QBrush(Qt.green))
            # 按列添加项进去(必须有)
            self.ui.tableWidget.setHorizontalHeaderItem(i, headerItem)
        # 2、设置 行数 和 行内容
        rowText = [
            "均值(mean)",
            "绝对平均值(absolute_mean)",
            "均方根(rms)",
            "方根幅值(sra)",
            "方差var",
            "标准差(std)",
            "最大值(max)",
            "最小值(min)",
            "偏度(skew)",
            "峭度(kurt)",
            "峰值(peak)",
            "裕度因子(margin_factor)",
            "波形因子(shape_factor)",
            "脉冲因子(impulse_factor)",
            "峰值因子(crest_factor)"]
        # 设置行数
        self.ui.tableWidget.setRowCount(len(rowText))
        # print(self.ui.tableWidget.rowCount())
        # 设置行内容 根据单元格类型指定赋值  单元格类型为枚举类型 需事先定义 自定义的枚举类型为1和2分别表示参数类型和参数大小
        # 默认初始化参数使用电流信号的220V
        features = self.calculate_waveform_characteristics("220",type="current")
        for rowIndex in range(self.ui.tableWidget.rowCount()):  # 每行开始循环
            # 1、指定创建的项为参数类型(即第1列)
            parameter_name_item = QTableWidgetItem(
                rowText[rowIndex], cellType.parameter_name_type.value)
            # 设置格式
            font = parameter_name_item.font()
            font.setBold(True)
            font.setPointSize(15)
            font.setFamily("楷体")
            parameter_name_item.setFont(font)
            parameter_name_item.setForeground(QBrush(Qt.red))
            parameter_name_item.setTextAlignment(
                Qt.AlignVCenter | Qt.AlignHCenter)
            # 关联数据(可以不关联,只是为了方便查询该单元格信息 可以显示在状态栏)
            parameter_name_item.setData(Qt.UserRole, rowText[rowIndex])
            # 添加项进入
            self.ui.tableWidget.setItem(
                rowIndex,
                columnIndex.parameter_name_num.value,
                parameter_name_item)
            # 2、指定创建的项为值类型(即第2列) 初始化默认值参数为220V的特征(要求字符型)
            parameter_value_item = QTableWidgetItem(
                str(features[rowIndex]), cellType.parameter_value_type.value)
            # 设置格式
            font.setFamily("TimeNewRoman")
            font.setBold(False)  # 值不加粗
            parameter_value_item.setFont(font)
            parameter_value_item.setForeground(QBrush(Qt.blue))
            parameter_value_item.setTextAlignment(
                Qt.AlignVCenter | Qt.AlignHCenter)
            # 关联数据(可以不关联,只是为了方便查询该单元格信息 可以显示在状态栏) 将当前项的文字进行存储
            parameter_value_item.setData(
                Qt.UserRole, parameter_value_item.text())
            # 添加项进入
            self.ui.tableWidget.setItem(
                rowIndex,
                columnIndex.parameter_value_num.value,
                parameter_value_item)
        # 最后一行信息对应正确说明初始化成功 否则失败
        # if self.ui.tableWidget.item( self.ui.tableWidget.rowCount()-1 , 0).text() =="峰值因子(crest_factor)" \
        #     and self.ui.tableWidget.item(self.ui.tableWidget.rowCount()-1 , 1 ).text() == "0":
        # 由于值单元格的值会更新不为0 所以不需要过强的判断 检测参数名字符合即可
        if self.ui.tableWidget.item(
                self.ui.tableWidget.rowCount() - 1,
                0).text() == "峰值因子(crest_factor)":
            print("波形时域特征参数初始化成功...")
            self._init_tableWidget = True
            return True  # 用于检查表格是否初始化成功
        else:
            print("波形时域特征初始化失败..")
            return False

    # 7、根据时域特征表格当前项的变化去改变状态栏信息
    # 使用currentCellChanged而不是currentItemChanged,后者需要项内容改变才发射信号
    @pyqtSlot(int, int, int, int)
    def on_tableWidget_currentCellChanged(self, currentRow, currentColumn,
                                          previousRow, previousColumn):
        # 参数 : 当前行列和前一行列的位置
        # 1、首先检查表格是否初始化
        if self._init_tableWidget:
            # 不要用下方的语句进行判断 否则又会初始化
            # if self.init_tableWidget_time_features() == False :
            #     return
            # 如果初始化过了直接显示状态栏信息
            self.ui.statusbar.clearMessage()
            currentItem = self.ui.tableWidget.item(currentRow, currentColumn)
            # print(currentRow,currentRow)
            userdata = currentItem.data(Qt.UserRole)  # 读取当前单元格存储的用户数据
            if currentItem.type() == cellType.parameter_name_type.value:
                self.ui.statusbar.showMessage("这是一个当前波形的时域特征参数 : " + userdata)
            elif currentItem.type() == cellType.parameter_value_type.value:
                # 如果是值类型单元格 ,找到当前项的左边也就是参数类型
                leftCurrentItem = self.ui.tableWidget.item(
                    currentRow, currentColumn - 1)
                leftuserdata = leftCurrentItem.data(Qt.UserRole)
                self.ui.statusbar.showMessage(
                    "这是一个当前波形的时域特征 : " + leftuserdata + " ,其大小为 : " + userdata)
            else:
                self.ui.statusbar.showMessage("当前波形的时域特征没有取值")
        else:
            # 没有初始化就重新初始化
            self.init_table_time_features()
    # 8、调用时域特征给表格赋值

    def set_table_values(self, type, option):
        # 如果只需要展示正常波形 用于界面初始化时显示
        if type == "电流":
            # 计算指定的option的波形数据
            features_data = self.calculate_waveform_characteristics(option,type="current")
            return features_data
        elif type == "振动":
            features_data = self.calculate_waveform_characteristics(option, type="vibration")
            return features_data
        else:
            print("类型type只能输入<电流>或<振动>！")
            return False

    # 9、初始化其他属性
    def init_other_properties(self):
        self.ui.mlp_btn.setEnabled(True)  # 两个btn都可选
        self.ui.svm_btn.setEnabled(True)
        self.ui.mlp_btn.setChecked(True)  # 默认使用MLP算法
        self.ui.svm_btn.setChecked(False)
        self.ui.maxiter_spinBox.setMinimum(1)  # 默认最小训练次数
        self.ui.maxiter_spinBox.setMaximum(10000)  # 最大训练次数10000
        self.ui.maxiter_spinBox.setValue(10)  # 默认训练10次
        data_num = self.read_current_waveform_pca_features(
        ).shape[0]  # 241 最大训练次数
        # print(data_num)
        self.ui.kfold_spinBox.setMaximum(data_num)  # 最大交叉验证次数为数据量的大小
        self.ui.kfold_spinBox.setMinimum(0)  # 交叉验证最小2 如果为0和1说明使用留出法而不是交叉验证
        self.ui.kfold_spinBox.setValue(0)  # 默认使用留出法
        self.ui.progressBar.setFormat("%p%")  # 百分比形式
        self.ui.progressBar.setMaximum(
            self.ui.maxiter_spinBox.value())  # 进度条 最大值为当前训练次数
        self.ui.progressBar.setMinimum(0)
        self.ui.progressBar.setValue(int(self.ui.maxiter_spinBox.value() / 2))
        # 初始化按钮图标
        self.ui.svm_btn.setIcon(QIcon(":/images/svm.jpg"))
        self.ui.svm_btn.setIconSize(QSize(64, 64))
        self.ui.mlp_btn.setIcon(QIcon(":/images/mlp.jpg"))
        self.ui.mlp_btn.setIconSize(QSize(64, 64))
        self.ui.start_training.setIcon(QIcon(":/images/TrainingIcon.ico"))
        self.ui.start_training.setIconSize(QSize(64, 64))
        self.ui.pushButton.setIcon(QIcon(":/images/LoadWaveform.ico"))
        self.ui.pushButton.setIconSize(QSize(96, 96))
        self.setWindowIcon(QIcon(":/images/platformIcon.ico"))
        self.setIconSize(QSize(128, 128))
        self.ui.cancel_training.setIcon(QIcon(":/images/cancel.png"))
        self.ui.cancel_training.setIconSize(QSize(64, 64))
        self.ui.close_action.setIcon(QIcon(":/images/close.ico"))
        self.ui.start_action.setIcon(QIcon(":/images/TrainingIcon.ico"))
        self.ui.cancel_action.setIcon(QIcon(":/images/cancel.png"))
        self.ui.load_action.setIcon(QIcon(":/images/LoadWaveform.ico"))
        # 设置菜单栏格式
        font = QFont()
        font.setFamily("楷体")
        font.setPointSize(15)
        self.ui.menubar.setFont(font)
        self.ui.comand_menu.setFont(font)
        self.ui.close_menu.setFont(font)
        # 设置窗口标题格式
        # font.setFamily("黑体")
        # font.setBold(True)
        # font.setPointSize(20)
        # self.setFont(font)
        # 设置窗口属性
        # 设置背景图片的操作
        # 加背景会导致一些按钮、状态栏看不到
        palette = QPalette()
        path = os.getcwd() + "/MyPlatform/background.jpg"
        pix = QPixmap(path)
        pix.scaled(self.width(), self.height())
        palette.setBrush(QPalette.Background, QBrush(pix))
        self.setPalette(palette)
        #self.setStyleSheet("background-image: url(:/images/platformBackground.jpg);")
        self.setWindowTitle("低压断路器故障诊断与在线监测界面")
        self.setIconSize(QSize(128, 128))
        # self.setWindowOpacity(0.9) # 窗口透明度
        self.setAttribute(Qt.WA_TranslucentBackground)  # 设置窗口背景透明
        # 有背景图片的话需要更改QLabel的字体颜色 否则看不见 (白色)
        self.ui.statusbar.setStyleSheet(
            "color:rgb({},{},{},255)".format(
                255, 255, 255))
        self.ui.maxiter_label.setStyleSheet(
            "color:rgb({},{},{},255)".format(255, 255, 255))
        self.ui.kfold_label.setStyleSheet(
            "color:rgb({},{},{},255)".format(
                255, 255, 255))
        self.ui.combox_label.setStyleSheet(
            "color:rgb({},{},{},255)".format(255, 255, 255))
        self.ui.progresser_label.setStyleSheet(
            "color:rgb({},{},{},255)".format(255, 255, 255))
        self.ui.predict_features_label.setStyleSheet(
            "color:rgb({},{},{},255)".format(255, 255, 255))
        self.ui.time_features_label.setStyleSheet(
            "color:rgb({},{},{},255)".format(255, 255, 255))
        self.ui.svm_btn.setStyleSheet(
            "color: rgb(255, 0, 0);background-color: rgb(0, 0, 0);"
            "alternate-background-color: qlineargradient(spread:pad, x1:0, "
            "y1:0, x2:1, y2:0, stop:0 rgba(255, 0, 0, 255), "
            "stop:0.166 rgba(255, 255, 0, 255), stop:0.333 rgba(0, 255, 0, 255), "
            "stop:0.5 rgba(0, 255, 255, 255), stop:0.666 rgba(0, 0, 255, 255), "
            "stop:0.833 rgba(255, 0, 255, 255), stop:1 rgba(255, 0, 0, 255));")
        self.ui.mlp_btn.setStyleSheet(
            "color: rgb(255, 0, 0);background-color: rgb(0, 0, 0);"
            "alternate-background-color: qlineargradient(spread:pad, x1:0, "
            "y1:0, x2:1, y2:0, stop:0 rgba(255, 0, 0, 255), "
            "stop:0.166 rgba(255, 255, 0, 255), stop:0.333 rgba(0, 255, 0, 255), "
            "stop:0.5 rgba(0, 255, 255, 255), stop:0.666 rgba(0, 0, 255, 255), "
            "stop:0.833 rgba(255, 0, 255, 255), stop:1 rgba(255, 0, 0, 255));")
        # 设置QLabel文字
        self.ui.kfold_label.setText("交叉验证次数(kfold)")
        self.ui.maxiter_label.setText("训练次数(maxiter)")
        self.ui.time_features_label.setText("波形的时域特征")
        self.ui.predict_features_label.setText("波形的特征预测")
        self.ui.progresser_label.setText("模型训练进度 : ")
        self.ui.combox_label.setText("请选择加载的波形 : ")

    # 10、初始化电流故障类型的表格 : 调用普通预测函数 不带maxiter和kfold
    def init_current_tableWidget(self):
        # 1、设置表头、列数 、是否可编辑、交替行背景、自动调整行高列宽
        self.ui.current_predict_tabWidget.setAlternatingRowColors(
            True)  # 交替行背景
        self.ui.current_predict_tabWidget.resizeRowsToContents()  # 行高
        # self.ui.current_predict_tabWidget.resizeColumnsToContents()  # 列宽
        self.ui.current_predict_tabWidget.setColumnWidth(
            columnIndex.parameter_name_num.value, 325)
        self.ui.current_predict_tabWidget.setColumnWidth(
            columnIndex.parameter_value_num.value, 325)
        headerText = ["故障类型", "故障概率"]
        self.ui.current_predict_tabWidget.setColumnCount(
            len(headerText))  # 表格列数
        self.ui.current_predict_tabWidget.setEditTriggers(
            QAbstractItemView.NoEditTriggers)  # 不可编辑
        for i in range(len(headerText)):
            #  设置表头
            headerItem = QTableWidgetItem(headerText[i])
            # 设置表头格式
            font = headerItem.font()
            font.setBold(True)
            font.setPointSize(15)
            font.setFamily("楷体")
            headerItem.setFont(font)
            # 设置对齐方式
            headerItem.setTextAlignment(Qt.AlignVCenter)
            headerItem.setTextAlignment(Qt.AlignHCenter)
            # 文字颜色
            headerItem.setForeground(QBrush(Qt.black))
            # 按列添加项进去(必须有)
            self.ui.current_predict_tabWidget.setHorizontalHeaderItem(
                i, headerItem)
            # 2、设置 行数 和 行内容
        rowText = ["线圈欠压160V", "线圈欠压180V", "线圈欠压200V", "线圈正常220V",
                   "线圈顶杆卡涩", "线圈轻微老化", "线圈严重老化", "线圈过压240V"]
        # 设置行数
        self.ui.current_predict_tabWidget.setRowCount(len(rowText))
        # print(self.ui.tableWidget.rowCount())
        # 设置行内容 根据单元格类型指定赋值  单元格类型为枚举类型 需事先定义 自定义的枚举类型为1和2分别表示参数类型和参数大小
        # 默认初始化参数使用电流信号的220V的预测概率
        # 14、对默认返回预测的mlp概率矩阵求平均
        y_predict_proba_mean = self.return_current_predict_mean_proba_mlp(
            "220")  # 返回数组格式
        # print(y_predict_proba_mean)
        for rowIndex in range(
                self.ui.current_predict_tabWidget.rowCount()):  # 每行开始循环
            # 1、指定创建的项为参数类型(即第1列)
            # print(rowIndex)
            parameter_name_item = QTableWidgetItem(
                rowText[rowIndex], cellType.parameter_name_type.value)
            # 设置格式
            font = parameter_name_item.font()
            font.setBold(True)
            font.setPointSize(15)
            font.setFamily("楷体")
            parameter_name_item.setFont(font)
            parameter_name_item.setForeground(QBrush(Qt.red))
            parameter_name_item.setTextAlignment(
                Qt.AlignVCenter | Qt.AlignHCenter)
            # 关联数据(可以不关联,只是为了方便查询该单元格信息 可以显示在状态栏)
            parameter_name_item.setData(Qt.UserRole, rowText[rowIndex])
            # 添加项进入
            self.ui.current_predict_tabWidget.setItem(
                rowIndex, columnIndex.parameter_name_num.value, parameter_name_item)
            # 2、指定创建的项为值类型(即第2列) 初始化默认值参数为220V的特征(要求字符型)
            parameter_value_item = QTableWidgetItem(str(
                round(y_predict_proba_mean[rowIndex], 5)), cellType.parameter_value_type.value)
            # 设置格式
            font.setFamily("TimeNewRoman")
            font.setBold(False)  # 值不加粗
            parameter_value_item.setFont(font)
            parameter_value_item.setForeground(QBrush(Qt.blue))
            parameter_value_item.setTextAlignment(
                Qt.AlignVCenter | Qt.AlignHCenter)
            # 关联数据(可以不关联,只是为了方便查询该单元格信息 可以显示在状态栏) 将当前项的文字进行存储
            parameter_value_item.setData(
                Qt.UserRole, parameter_value_item.text())
            # 添加项进入
            self.ui.current_predict_tabWidget.setItem(
                rowIndex, columnIndex.parameter_value_num.value, parameter_value_item)
            # 最后一行信息对应正确说明初始化成功 否则失败
        if self.ui.current_predict_tabWidget.item(
                self.ui.current_predict_tabWidget.rowCount() - 1,
                0).text() == "线圈过压240V":
            print("电流波形故障预测概率初始化成功...")
            self._init_current_tableWidget = True
            return True  # 用于检查表格是否初始化成功
        else:
            print("电流波形故障预测概率初始化失败..")
            return False

    # 11、初始化振动故障类型的表格 : 调用普通预测函数 不带maxiter和kfold
    def init_vibration_tableWidget(self):
        # 1、设置表头、列数 、是否可编辑、交替行背景、自动调整行高列宽
        self.ui.vbriation_predict_tabWidget.setAlternatingRowColors(
            True)  # 交替行背景
        self.ui.vbriation_predict_tabWidget.resizeRowsToContents()  # 行高
        # self.ui.vbriation_predict_tabWidget.resizeColumnsToContents()  # 列宽
        self.ui.vbriation_predict_tabWidget.setColumnWidth(
            columnIndex.parameter_name_num.value, 325)
        self.ui.vbriation_predict_tabWidget.setColumnWidth(
            columnIndex.parameter_value_num.value, 325)
        headerText = ["故障类型", "故障概率"]
        self.ui.vbriation_predict_tabWidget.setColumnCount(
            len(headerText))  # 表格列数
        self.ui.vbriation_predict_tabWidget.setEditTriggers(
            QAbstractItemView.NoEditTriggers)  # 不可编辑
        for i in range(len(headerText)):
            #  设置表头
            headerItem = QTableWidgetItem(headerText[i])
            # 设置表头格式
            font = headerItem.font()
            font.setBold(True)
            font.setPointSize(15)
            font.setFamily("楷体")
            headerItem.setFont(font)
            # 设置对齐方式
            headerItem.setTextAlignment(Qt.AlignVCenter)
            headerItem.setTextAlignment(Qt.AlignHCenter)
            # 文字颜色
            headerItem.setForeground(QBrush(Qt.black))
            # 按列添加项进去(必须有)
            self.ui.vbriation_predict_tabWidget.setHorizontalHeaderItem(
                i, headerItem)
        # 2、设置 行数 和 行内容
        rowText = ["正常合闸", "正常分闸","虚假合闸","分闸不彻底",
                   "储能弹簧卡涩", "操作机构卡涩","合闸螺栓松动", "分闸螺栓松动"]
        # 设置行数
        self.ui.vbriation_predict_tabWidget.setRowCount(len(rowText))
        # print(self.ui.tableWidget.rowCount())
        # 设置行内容 根据单元格类型指定赋值  单元格类型为枚举类型 需事先定义 自定义的枚举类型为1和2分别表示参数类型和参数大小
        # 默认初始化参数使用振动信号的正常分合闸的预测概率
        y_predict_proba_mean = self.return_vibration_predict_mean_proba_mlp(
            "正常合闸")  # 返回数组格式
        # y = [6, 5, 1, 7, 0, 4, 3, 2]
        # 调整顺序,同时交换1和7--->虚假合闸和分闸不彻底交换
        adjust_sequence_y_predict_proba_mean = [y_predict_proba_mean[6], y_predict_proba_mean[5],
                                                y_predict_proba_mean[7], y_predict_proba_mean[1],
                                                y_predict_proba_mean[0], y_predict_proba_mean[4],
                                                y_predict_proba_mean[3], y_predict_proba_mean[2]]
        for rowIndex in range(
                self.ui.vbriation_predict_tabWidget.rowCount()):  # 每行开始循环
            # 1、指定创建的项为参数类型(即第1列)
            parameter_name_item = QTableWidgetItem(
                rowText[rowIndex], cellType.parameter_name_type.value)
            # 设置格式
            font = parameter_name_item.font()
            font.setBold(True)
            font.setPointSize(15)
            font.setFamily("楷体")
            parameter_name_item.setFont(font)
            parameter_name_item.setForeground(QBrush(Qt.red))
            parameter_name_item.setTextAlignment(
                Qt.AlignVCenter | Qt.AlignHCenter)
            # 关联数据(可以不关联,只是为了方便查询该单元格信息 可以显示在状态栏)
            parameter_name_item.setData(Qt.UserRole, rowText[rowIndex])
            # 添加项进入
            self.ui.vbriation_predict_tabWidget.setItem(
                rowIndex, columnIndex.parameter_name_num.value, parameter_name_item)
            # print(self.ui.vbriation_predict_tabWidget.item(
            #         rowIndex, columnIndex.parameter_name_num.value).text())
            # 2、指定创建的项为值类型(即第2列) 初始化默认值参数为正常合闸的特征(要求字符型)
            # y = [6, 5, 1, 7, 0, 4, 3, 2]
            #print(round(adjust_sequence_y_predict_proba_mean[rowIndex], 5))
            parameter_value_item = QTableWidgetItem(str(
                round(adjust_sequence_y_predict_proba_mean[rowIndex], 5)), cellType.parameter_value_type.value)
            # 设置格式
            font.setFamily("TimeNewRoman")
            font.setBold(False)  # 值不加粗
            parameter_value_item.setFont(font)
            parameter_value_item.setForeground(QBrush(Qt.blue))
            parameter_value_item.setTextAlignment(
                Qt.AlignVCenter | Qt.AlignHCenter)
            # 关联数据(可以不关联,只是为了方便查询该单元格信息 可以显示在状态栏) 将当前项的文字进行存储
            parameter_value_item.setData(
                Qt.UserRole, parameter_value_item.text())
            # 添加项进入
            self.ui.vbriation_predict_tabWidget.setItem(
                rowIndex, columnIndex.parameter_value_num.value, parameter_value_item)
            #print(self.ui.vbriation_predict_tabWidget.item(rowIndex, columnIndex.parameter_value_num.value).text())
            # 最后一行信息对应正确说明初始化成功 否则失败
        if self.ui.vbriation_predict_tabWidget.item(
                self.ui.vbriation_predict_tabWidget.rowCount() - 1,
                0).text() == "分闸螺栓松动":
            print("振动波形故障预测概率初始化成功...")
            self._init_vibration_tableWidget = True
            return True  # 用于检查表格是否初始化成功
        else:
            print("振动波形故障预测概率初始化失败..")
            return False

    # 12、根据电流故障概率当前项的变化去改变状态栏信息
    @pyqtSlot(int, int, int, int)
    # 使用currentCellChanged而不是currentItemChanged,后者需要项内容改变才发射信号
    def on_current_predict_tabWidget_currentCellChanged(
            self, currentRow, currentColumn, previousRow, previousColumn):
        # 参数 : 当前行列和前一行列的位置
        # 1、首先检查表格是否初始化
        if self._init_current_tableWidget:
            # 如果初始化过了直接显示状态栏信息
            self.ui.statusbar.clearMessage()
            currentItem = self.ui.current_predict_tabWidget.item(
                currentRow, currentColumn)
            # print(currentRow,currentRow)
            userdata = currentItem.data(Qt.UserRole)  # 读取当前单元格存储的用户数据
            # print(userdata)
            if currentItem.type() == cellType.parameter_name_type.value:
                self.ui.statusbar.showMessage("这是一个当前电流波形的预测类型 : " + userdata)
            elif currentItem.type() == cellType.parameter_value_type.value:
                # 如果是值类型单元格 ,找到当前项的左边也就是参数类型
                leftCurrentItem = self.ui.current_predict_tabWidget.item(
                    currentRow, currentColumn - 1)
                leftuserdata = leftCurrentItem.data(Qt.UserRole)
                self.ui.statusbar.showMessage(
                    "这是一个当前电流波形的预测类型 : " + leftuserdata + " ,其概率为 : " + userdata)
            else:
                self.ui.statusbar.showMessage("当前电流波形的预测概率没有取值")
        else:
            # 没有初始化就重新初始化
            self.init_current_tableWidget()

    # 13、根据振动故障概率当前项的变化去改变状态栏信息
    # 使用currentCellChanged而不是currentItemChanged,后者需要项内容改变才发射信号
    @pyqtSlot(int, int, int, int)
    def on_vbriation_predict_tabWidget_currentCellChanged(
            self, currentRow, currentColumn, previousRow, previousColumn):
        # 参数 : 当前行列和前一行列的位置
        # 1、首先检查表格是否初始化
        if self._init_vibration_tableWidget:
            # 如果初始化过了直接显示状态栏信息
            self.ui.statusbar.clearMessage()
            currentItem = self.ui.vbriation_predict_tabWidget.item(
                currentRow, currentColumn)
            # print(currentRow,currentRow)
            userdata = currentItem.data(Qt.UserRole)  # 读取当前单元格存储的用户数据
            # print(userdata)
            if currentItem.type() == cellType.parameter_name_type.value:
                self.ui.statusbar.showMessage("这是一个当前振动波形的预测类型 : " + userdata)
            elif currentItem.type() == cellType.parameter_value_type.value:
                # 如果是值类型单元格 ,找到当前项的左边也就是参数类型
                leftCurrentItem = self.ui.vbriation_predict_tabWidget.item(
                    currentRow, currentColumn - 1)
                leftuserdata = leftCurrentItem.data(Qt.UserRole)
                self.ui.statusbar.showMessage(
                    "这是一个当前振动波形的预测类型 : " + leftuserdata + " ,其概率为 : " + userdata)
            else:
                self.ui.statusbar.showMessage("当前振动波形的预测概率没有取值")
        else:
            # 没有初始化就重新初始化
            self.init_vibration_tableWidget()

    # 14、调用svm算法执行
    def start_svm_training(self):
        # 当按钮为MLP时执行 第一步先找到option 这个根据comboBox的选择确定
        comboBox_item_int = self.ui.comboBox.currentIndex()  # 整型
        if self.convert_faultNumber_intToString(comboBox_item_int):
            comboBox_item_str = self.convert_faultNumber_intToString(
                comboBox_item_int)
            # 得到代号后继续判断是振动还是电流
            if comboBox_item_int < 8:
                # 1、 编号在0~7为电流信号
                #  1.1、计算电流信号指定option的预测概率
                # 调用时需要指定模型类型svm、训练次数maxiter、迭代次数kfold、指定训练波形option
                maxiter = self.ui.maxiter_spinBox.value()
                kfold = self.ui.kfold_spinBox.value()
                y_predict_proba_mean, usetime = self.return_current_model_traingtime(
                    "svm", maxiter, kfold, comboBox_item_str)
                self.ui.statusbar.showMessage("波形预测概率加载成功...", 10000)
                # 1.2、将概率传递给表格
                # 将这些参数传递给项第2列
                for index, value in enumerate(y_predict_proba_mean):
                    # index = 0~7 也作为索引项的行号 列号不变 然后将value也就是波形特征数据直接设置为值
                    # print(index,value)

                    self.ui.current_predict_tabWidget.item(
                        index, columnIndex.parameter_value_num.value).setText(str(round(value, 5)))
                    # print(self.ui.tableWidget.item(index,columnIndex.parameter_value_num.value).text())
                    # 1.3、同时更新用户数据
                    self.ui.current_predict_tabWidget.item(
                        index, columnIndex.parameter_value_num.value).setData(Qt.UserRole, str(round(value, 5)))
                    # print(self.ui.tableWidget.item(index, columnIndex.parameter_value_num.value).data(Qt.UserRole))
                # 1.4、更新状态栏信息(时间、故障类型和概率)
                self.ui.statusbar.clearMessage()
                # time.sleep(2)
                # 1.4.1、找到故障类型 利用comboBox_item_int作为行获取存储在表格的用户数据
                fault_type = self.ui.current_predict_tabWidget.item(
                    comboBox_item_int, columnIndex.parameter_name_num.value).text()
                #print(fault_type)
                # 1.4.2、找到故障概率
                fault_proba = self.ui.current_predict_tabWidget.item(
                    comboBox_item_int, columnIndex.parameter_value_num.value).text()
                #print(fault_proba)
                # 1.4.3、状态栏打印
                self.ui.statusbar.showMessage("模型训练耗时时间为 : " + usetime + " s" +
                                              " 预测的故障类型为 : " + fault_type
                                              + " 其故障概率为 : " + fault_proba)
                # # 1.5、进度条函数
                # counter = QTime() # 计时器
                # for i in range(self.ui.progressBar.maximum() + 1) :
                #     self.ui.progressBar.setValue(i)
                #     counter.start()
                # self._init_current_tableWidget = True  # 防止选取不同波形时重复初始化

            else:
                # 2、编号8~15为振动信号 功能后续添加
                #  2.1、计算振动信号指定option的预测概率
                # 调用时需要指定模型类型svm、训练次数maxiter、迭代次数kfold、指定训练波形option
                maxiter = self.ui.maxiter_spinBox.value()
                kfold = self.ui.kfold_spinBox.value()
                y_predict_proba_mean, usetime = self.return_vibration_model_traingtime(
                    "svm", maxiter, kfold, comboBox_item_str)
                # 由于振动信号标签不是按顺序分 而是按照6,5,1,7,0,4,3,2进行但是这个标签是按照vibration_pca_features_data.csv
                # 所以还需要交换虚假合闸和分闸不彻底才是对应的概率 即交换 1和7
                # 所以合闸其实是y_predict_proba_mean的第6个元素
                # 所以还需要调整概率一一对应的顺序
                adjust_sequence_y_predict_proba_mean = [y_predict_proba_mean[6],y_predict_proba_mean[5],
                                                        y_predict_proba_mean[7],y_predict_proba_mean[1],
                                                        y_predict_proba_mean[0],y_predict_proba_mean[4],
                                                        y_predict_proba_mean[3],y_predict_proba_mean[2]]

                #print(y_predict_proba_mean)
                # print(comboBox_item_str)
                self.ui.statusbar.showMessage("波形预测概率加载成功...", 10000)
                # 2.2、将概率传递给表格
                # 将这些参数传递给项第2列
                for index, value in enumerate(adjust_sequence_y_predict_proba_mean):
                # index = 8~15 也作为索引项的行号 列号不变 然后将value也就是波形特征数据直接设置为值
                    #print(index,value) # index=0~7 value=(proba0~proba7) 按照表格的正常顺序
                    self.ui.vbriation_predict_tabWidget.item(
                         index, columnIndex.parameter_value_num.value).setText(str(round(value, 5)))
                    # 2.3、同时更新用户数据
                    self.ui.vbriation_predict_tabWidget.item(
                        index, columnIndex.parameter_value_num.value).setData(Qt.UserRole, str(round(value, 5)))
                # 2.4、更新状态栏信息(时间、故障类型和概率)
                self.ui.statusbar.clearMessage()
                # time.sleep(2)
                # 2.4.1、找到故障类型 利用comboBox_item_int作为行获取存储在表格的用户数据
                #print(comboBox_item_int, columnIndex.parameter_name_num.value)
                # 对于comboBox_item_int虽然是8到15 但是对vbriation_predict_tabWidget仍应该访问0~7的
                fault_type = self.ui.vbriation_predict_tabWidget.item(
                    comboBox_item_int -8, columnIndex.parameter_name_num.value).text()
                #print(fault_type)
                # 2.4.2、找到故障概率
                fault_proba = self.ui.vbriation_predict_tabWidget.item(
                    comboBox_item_int -8, columnIndex.parameter_value_num.value).text()
                #print(fault_proba)
                # 2.4.3、状态栏打印
                self.ui.statusbar.showMessage("模型训练耗时时间为 : " + usetime + " s" +
                                              " 预测的故障类型为 : " + fault_type
                                              + " 其故障概率为 : " + fault_proba)
        else:  # 转换整型失败时候返回
            self.ui.statusbar.clearMessage()
            self.ui.statusbar.showMessage("波形预测概率加载失败...")

    # 15、调用mlp算法执行
    def start_mlp_training(self):
        # 当按钮为MLP时执行 第一步先找到option 这个根据comboBox的选择确定
        comboBox_item_int = self.ui.comboBox.currentIndex()  # 整型
        if self.convert_faultNumber_intToString(comboBox_item_int):
            comboBox_item_str = self.convert_faultNumber_intToString(
                comboBox_item_int)
            # 得到代号后继续判断是振动还是电流
            if comboBox_item_int < 8:
                # 1、 编号在0~7为电流信号
                #  计算电流信号指定option的预测概率
                # 调用时需要指定模型类型svm、训练次数maxiter、迭代次数kfold、指定训练波形option
                maxiter = self.ui.maxiter_spinBox.value()
                kfold = self.ui.kfold_spinBox.value()
                y_predict_proba_mean, usetime = self.return_current_model_traingtime(
                    "mlp", maxiter, kfold, comboBox_item_str)
                # print(y_predict_proba_mean,type(y_predict_proba_mean))
                self.ui.statusbar.showMessage("波形预测概率加载成功...", 10000)
                # 将概率传递给表格
                # 将这些参数传递给项第2列
                for index, value in enumerate(y_predict_proba_mean):
                    # index = 0~7 也作为索引项的行号 列号不变 然后将value也就是波形特征数据直接设置为值
                    # print(index,value)

                    self.ui.current_predict_tabWidget.item(
                        index, columnIndex.parameter_value_num.value).setText(str(round(value, 5)))
                    # print(self.ui.tableWidget.item(index,columnIndex.parameter_value_num.value).text())
                    # 同时更新用户数据
                    self.ui.current_predict_tabWidget.item(
                        index, columnIndex.parameter_value_num.value).setData(Qt.UserRole, str(round(value, 5)))
                    # print(self.ui.tableWidget.item(index, columnIndex.parameter_value_num.value).data(Qt.UserRole))
                # 1.1、更新状态栏信息(时间、故障类型和概率)
                self.ui.statusbar.clearMessage()
                # time.sleep(2)
                fault_type = self.ui.current_predict_tabWidget.item(
                    comboBox_item_int, columnIndex.parameter_name_num.value).text()
                # 1.1.1、找到故障概率
                fault_proba = self.ui.current_predict_tabWidget.item(
                    comboBox_item_int, columnIndex.parameter_value_num.value).text()
                # 1.1.2、状态栏打印
                self.ui.statusbar.showMessage("模型训练耗时时间为 : " + usetime + " s" +
                                              " 预测的故障类型为 : " + fault_type
                                              + " 其故障概率为 : " + fault_proba)
                # # 1.5、进度条函数
                # counter = QTime() # 计时器
                # for i in range(self.ui.progressBar.maximum() + 1) :
                #     self.ui.progressBar.setValue(i)
                #     counter.start()
                # self._init_current_tableWidget = True  # 防止选取不同波形时重复初始化
            else:
                # 2、编号8~15为振动信号 功能后续添加
                #  计算电流信号指定option的预测概率
                # 调用时需要指定模型类型svm、训练次数maxiter、迭代次数kfold、指定训练波形option
                maxiter = self.ui.maxiter_spinBox.value()
                kfold = self.ui.kfold_spinBox.value()
                y_predict_proba_mean, usetime = self.return_vibration_model_traingtime(
                    "mlp", maxiter, kfold, comboBox_item_str)
                self.ui.statusbar.showMessage("波形预测概率加载成功...", 10000)
                # 由于振动信号标签不是按顺序分 而是按照6,5,1,7,0,4,3,2进行 但是这个标签是按照vibration_pca_features_data.csv
                # 所以还需要交换虚假合闸和分闸不彻底才是对应的概率 即交换 1和7
                # 所以合闸其实是y_predict_proba_mean的第6个元素
                # 所以还需要调整概率一一对应的顺序
                adjust_sequence_y_predict_proba_mean = [y_predict_proba_mean[6], y_predict_proba_mean[5],
                                                        y_predict_proba_mean[7], y_predict_proba_mean[1],
                                                        y_predict_proba_mean[0], y_predict_proba_mean[4],
                                                        y_predict_proba_mean[3], y_predict_proba_mean[2]]
                # 将概率传递给表格
                # 将这些参数传递给项第2列
                for index, value in enumerate(adjust_sequence_y_predict_proba_mean):
                    # index = 8~15 也作为索引项的行号 列号不变 然后将value也就是波形特征数据直接设置为值
                    # print(index,value)

                    self.ui.vbriation_predict_tabWidget.item(
                        index, columnIndex.parameter_value_num.value).setText(str(round(value, 5)))
                    # print(self.ui.tableWidget.item(index,columnIndex.parameter_value_num.value).text())
                    # 同时更新用户数据
                    self.ui.vbriation_predict_tabWidget.item(
                        index, columnIndex.parameter_value_num.value).setData(Qt.UserRole, str(round(value, 5)))
                # 2.1、更新状态栏信息(时间、故障类型和概率)
                self.ui.statusbar.clearMessage()
                # time.sleep(2)
                fault_type = self.ui.vbriation_predict_tabWidget.item(
                    comboBox_item_int-8, columnIndex.parameter_name_num.value).text()
                # 2.1.1、找到故障概率
                fault_proba = self.ui.vbriation_predict_tabWidget.item(
                    comboBox_item_int-8, columnIndex.parameter_value_num.value).text()
                # 2.1.2、状态栏打印
                self.ui.statusbar.showMessage("模型训练耗时时间为 : " + usetime + " s" +
                                              " 预测的故障类型为 : " + fault_type
                                        + " 其故障概率为 : " + fault_proba)
        else:  # 转换整型失败时候返回
            self.ui.statusbar.clearMessage()
            self.ui.statusbar.showMessage("波形预测概率加载失败...")

    @pyqtSlot(bool)
    # 16、开始训练函数(调用带maxiter和kfold的函数)
    def on_start_training_clicked(self, checked):
        # 点击训练时先设置取消训练没有动作标志位为假 即在训练状态
        # 结束训练后必须将状态重新更改 为False 这条语句在点击训练时实现了 被重置
        # 但是要注意的问题是 虽然被重置但是只有在点击重新训练时才被重置 这个状态可能存在风险
        # 这个风险可以在训练结束时就进行重置 避免此风险
        self.have_trained_model = True
        self.cancel_training_flag = False
        # 只要点击过按钮就可以将菜单初始化动作使能
        if self.have_loaded_waveform or self.have_trained_model:
            self.init_action.setEnabled(True)  # 禁用动作
            self.init_action.setChecked(True)
        else:
            self.init_action.setEnabled(False)  # 禁用动作
            self.init_action.setChecked(False)
        # 检测radioButton的checked状态
        radio_button_state = self.ui.svm_btn.isChecked()
        if radio_button_state:  # 选的svm算法
            # 保险一步操作 : 当svm按钮为真时禁用mlp状态
            self.ui.mlp_btn.setChecked(False)
            # 调用svm算法
            self.start_svm_training()
        elif radio_button_state == False:  # 选的mlp算法
            # 保险一步操作 : 当svm按钮为假时禁用svm状态
            self.ui.svm_btn.setChecked(False)
            # 调用mlp算法
            self.start_mlp_training()

    @pyqtSlot(bool)
    # 17、设置标志位为真 那么当前处于训练中时扫描到就会退出
    def on_cancel_training_clicked(self, checked):
        self.cancel_training_flag = True  # 点击了取消后就将 当前属性改为真 那么训练中扫描到会推出

    # 18、初始化菜单动作与按钮的联系
    def init_menuAction_buttonClicked(self):
        # 可以直接载ui文件建立槽机制也可以初始化时建立槽机制
        # 动作可以ui先创建 也可以QAction动态创建
        self.ui.cancel_action.triggered.connect(
            self.ui.cancel_training.clicked)
        self.ui.load_action.triggered.connect(self.ui.pushButton.clicked)
        self.ui.close_action.triggered.connect(self.close)
        self.ui.start_action.triggered.connect(self.ui.start_training.clicked)

    # 19、动态添加菜单栏和动作(初始化) 并将动作连接到槽函数
    def init_widget_menu(self):
        # 动态创建菜单
        menubar = self.menuBar()
        newMenu = menubar.addMenu("重载")
        # 动态创建动作
        font = QFont()
        font.setFamily("楷体")
        font.setPointSize(15)
        # action = QAction() 不能直接添加动作
        # 不局部初始化
        # action = QAction(QIcon(":/images/refresh.ico"),"初始化窗口",self)
        # action.setIconVisibleInMenu(True)
        # action.setFont(font)
        # 第一次动态创建时设置禁用
        # action.setEnabled(False) #禁用动作
        # action.setChecked(False)

        # 取消训练按钮是否点击的属性 self.cancel_training_flag == False
        # 此属性的目的是 : 当取消训练按钮 点击时 该属性变为真 然后在训练函数中每次训练都会重复检测该属性
        # 那么只要点击了取消 训练检测到当前取消训练属性为真True时 就会停止训练返回 并重新设置取消训练属性为False
        # 而本菜单可初始化动作的时机是检测到训练已经进行过 或者加载过波形 才会让初始化动作可用
        # 界面只有加载波形、开始训练可以改变界面状态 所以这两个用作判据 取消训练不作为判据

        # 训练结束时就已经被重置为False 所以在菜单的动作看来 这个属性一直都是False 不能用来作为标记 训练是否发生过
        # 可以专门定义新属性 只要点击过开始训练按钮 就将此属性设为真 此时才会开放菜单的禁用 可以重新初始化
        # if self.have_loaded_waveform == True or self.have_trained_model == True :
        #     action.setEnabled(True)  # 禁用动作
        #     action.setChecked(True)
        # else :
        #     action.setEnabled(False)  # 禁用动作
        #     action.setChecked(False)

        # 只有上述语句还不行 因为在点击了加载波形或者开始训练 虽然两个属性发生了变化
        # 但是本函数init_widget_menu()没有重新初始化 所以这两条语句还需要在两个按钮处添加
        # 然而此菜单动作是动态添加的 所以无法在两个按钮处调用self.ui的实体动作进行设置
        # 解决方法是在两个按钮处重新调用 init_widget_menu()函数 因为按钮已经将两个属性改变了
        # 那么就可以将动作设为可见 但是问题在于 此时动作是重新创建了1个 而不是原来的动作

        # 最终解决方案 : 为了能够找到此动态的菜单动作 应当在__inin__去初始化 而不是在局部函数init_widget_menu去定义
        # 这样就可以在点击按钮的click函数中去使用这个动态的按钮 进而设置使能

        self.init_action.setIconVisibleInMenu(True)
        self.init_action.setFont(font)
        self.init_action.setEnabled(False)
        self.init_action.setChecked(False)
        self.init_action.setShortcut('ctrl+shift+i')
        if self.have_loaded_waveform or self.have_trained_model:
            self.init_action.setEnabled(True)  # 启用动作
            self.init_action.setChecked(True)
        else:
            self.init_action.setEnabled(False)  # 禁用动作
            self.init_action.setChecked(False)
        # 只有上述语句还不行 因为在点击了加载波形或者开始训练 虽然两个属性发生了变化
        # 但是本函数init_widget_menu()没有重新初始化 所以这两条语句还需要在两个按钮处添加
        self.init_action.triggered.connect(self.init_myplatform)
        # # 菜单添加动作
        newMenu.addAction(self.init_action)

    # 20、动态动作的槽函数 : 重新初始化
    def init_myplatform(self):
        # 此行语句测试是为了知道triggered是否直接触发 即连接就触发
        # 结论是不会触发 菜单点击才会触发 如果不能触发就无法利用槽函数来给菜单动作设置使能
        # 因为如果可以直接触发 就可以在槽函数先检测两个属性 然后设置使能
        # print("123")
        # 仍然存在一个问题就是重新初始化后 产生会一个新窗口实例
        # 因为__init__()会产生一个新实例
        self.__del__()  # 先析构掉当前窗口
        self.__init__()

    # 21、析构函数
    def __del__(self):
        self.close()

    @pyqtSlot(bool, bool)
    # 22、窗口类登陆界面的槽函数
    def signWindowIsSuccess(self, usernameIsTrue, passwordIsTrue):
        #print(usernameIsTrue, passwordIsTrue)
        if usernameIsTrue and passwordIsTrue:
            # print(True)
            self.mySignWindow.__del__()
            dialog_title = "登录成功消息框"
            dialog_information = "登陆成功,欢迎使用！"
            QMessageBox.information(self, dialog_title, dialog_information)
            self.ui.statusbar.showMessage("欢迎使用")
            self.show()
            return True
        else:
            # print(False)
            dialog_title = "登录失败消息框"
            dialog_information = "您的用户名或密码输入错误,请重新输入！"
            QMessageBox.warning(self, dialog_title, dialog_information)
            self.mySignWindow.__del__()
            # 不能使用 self.mySignWindow.__init__()产生新窗口
            # 因为这个新窗口不是具体的实例 没有与按钮动作绑定 会导致新出来的窗口点按钮点不动
            self.mySignWindow = mySignWindow(self)
            self.mySignWindow.signState.connect(self.signWindowIsSuccess)
            self.mySignWindow.ui.statusbar.showMessage("您的用户名或密码错误,请重新输入！")
            return False

    # 23、对话框类登陆界面的槽函数
    @pyqtSlot(bool, bool)
    def signDialogIsSuccess(self, usernameIsTrue, passwordIsTrue):
        print(usernameIsTrue, passwordIsTrue)
        if usernameIsTrue and passwordIsTrue:
            # print(True)
            self.mySignDialog.__del__()
            self.ui.statusbar.showMessage("欢迎使用")
            self.show()
            return True
        else:
            # print(False)
            # 对话框类没有属性状态栏 只能使用消息框 QMessageBox
            # self.mySignDialog.ui.statusbar.showMessage("您的用户名或密码错误,请重新输入！")
            dialog_title = "登录界面消息框"
            dialog_information = "您的用户名或密码输入错误,请重新输入！"
            QMessageBox.warning(self, dialog_title, dialog_information)
            self.mySignDialog.__del__()
            # 不能使用 self.mySignWindow.__init__()产生新窗口
            # 因为这个新窗口不是具体的实例 没有与按钮动作绑定 会导致新出来的窗口点按钮点不动
            self.mySignDialog = mySignDialog(self)
            self.mySignDialog.signState.connect(self.signDialogIsSuccess)
            return False

    '''2、界面用到的后台函数'''
    # 24、读取电流波形数据的函数
    def read_current_waveform_data(self):
        ''':return dataframe 1000×8'''
        path = os.getcwd() + "/MyPlatform/current_waveform_data.csv"
        # print(path)
        waveform_data = pd.read_csv(path, encoding='gbk')
        return waveform_data

    # 25、拆分电流波形数据得到8种波形的函数
    def spilit_current_waveform_data(self, option):
        ''':arg : string(160,180,200,220,240,stuck,age_10,age_50)
        :return : 某个波形的series'''
        waveform_data = self.read_current_waveform_data()
        keys = ["160", "180", "200", "220", "age_10", "age_50", "stuck", "240"]
        dict_waveform_data = {}
        for index, column in enumerate(waveform_data.columns):
            dict_waveform_data.update({keys[index]: waveform_data[column]})
        return dict_waveform_data[option]

    # 26、将故障的整型编号转换成字符串编号
    def convert_faultNumber_intToString(self, intNum):
        intNums = [x for x in range(self.ui.comboBox.count())]  # 故障编号0~15
        strNums = [
            "160","180","200","220",
            "stuck","age_10","age_50","240",
            "正常合闸","正常分闸","虚假合闸","分闸不彻底",
            "储能弹簧卡涩","操作机构卡涩","合闸螺栓松动","分闸螺栓松动"]
        for index, intnum in enumerate(intNums):
            # 传入的整型编号如果和intNums某个匹配,则返回对应的字符串代号
            if intNum == intnum:
                return strNums[index]
        return 0

    # 27、计算电流和振动波形时域特征
    def calculate_waveform_characteristics(self, option,type):
        ''':arg : 选择哪类波形 :type 选择电流或振动 :return 波形特征 list类型
        :internet https://blog.csdn.net/baidu_38963740/article/details/110940823'''
        if  type == "current" :
            option_waveform_data = self.spilit_current_waveform_data(
                option).values  # 转换成数组格式
        elif type == "vibration" :
            option_waveform_data = self.spilit_vibration_waveform_data(
                option).values  # 转换成数组格式
        else:
            print("选择的波形类型错误,type只有current或vibratuon选项！")
            return False
        # 1、平均值
        waveform_mean = np.mean(option_waveform_data)
        # 2、绝对平均值
        waveform_absolute_mean = waveform_mean / len(option_waveform_data)
        # 3、均方根值 = sqrt(Σxi^2 / N)
        waveform_rms = math.sqrt(
            sum([x ** 2 for x in option_waveform_data]) / len(option_waveform_data))
        # 4、方根幅值 : 所有点取绝对值再开根号 求和 求和后除长度N 之后再平方
        waveform_sra = pow(sum(
            [math.sqrt(abs(x)) for x in option_waveform_data]) / len(option_waveform_data), 2)
        # 5、方差
        waveform_var = np.var(option_waveform_data)
        # 6、标准差
        waveform_std = np.std(option_waveform_data)
        # 7、最大值
        waveform_max = np.max(option_waveform_data)
        # 8、最小值
        waveform_min = np.min(option_waveform_data)
        # 9、偏度
        waveform_skew = np.mean((option_waveform_data - waveform_mean) ** 3) \
            / pow(waveform_std, 3)
        # 10、峭度
        waveform_kurt = np.mean((option_waveform_data - waveform_mean) ** 4) \
            / pow(np.var(option_waveform_data), 2)
        # 11、峰值
        waveform_peak = max(abs(waveform_max), abs(waveform_min))
        # 12、裕度因子 = 峰值 / 方根幅值
        waveform_margin_factor = waveform_peak / waveform_sra
        # 13、波形因子 = 均方根值 / 均值
        waveform_shape_factor = waveform_rms / waveform_mean
        # 14、脉冲因子 = 峰值 / 均值
        waveform_impluse_factor = waveform_peak / waveform_mean
        # 15、峰值因子 = 峰值 / 均方根值
        waveform_crest_factor = waveform_peak / waveform_rms
        waveform_features = [
            round(
                waveform_mean, 3), round(
                waveform_absolute_mean, 3), round(
                waveform_rms, 3), round(
                    waveform_sra, 3), round(
                        waveform_var, 3), round(
                            waveform_std, 3), round(
                                waveform_max, 3), round(
                                    waveform_min, 3), round(
                                        waveform_skew, 3), round(
                                            waveform_kurt, 3), round(
                                                waveform_peak, 3), round(
                                                    waveform_margin_factor, 3), round(
                                                        waveform_shape_factor, 3), round(
                                                            waveform_impluse_factor, 3), round(
                                                                waveform_crest_factor, 3)]
        return waveform_features

    # 28、读取电流pca数据特征的函数
    def read_current_waveform_pca_features(self):
        path = os.getcwd() + "/MyPlatform/current_pca_features_data.csv"
        pca_features_data = pd.read_csv(path, encoding="gbk")
        return pca_features_data  # 241×2

    # 29、分离电流pca数据的函数
    def split_current_pca_features_data_label(self):
        pca_features_data = self.read_current_waveform_pca_features()
        X_dataframe = pca_features_data.iloc[:, 0:-1]  # 分出数据和标签 此时是DataFrame格式
        y_dataframe = pca_features_data.iloc[:, -1]  # 第3列是label
        X = X_dataframe.values  # ndarray格式 样本数×维数
        y_category = y_dataframe.values  # ndarray格式
        Label = LabelEncoder()  # 初始化1个独热编码类
        y = Label.fit_transform(y_category)  # 自动生成标签 不一定按顺序标 本文的数据是标签4和7互换
        return X, y

    # 30、继续分离电流信号的pca特征的数据得到8种故障类型的pca用于模型预测
    def spilit_current_pca_features_data(self, option):
        # pca的数据分离顺序是符合keys的顺序
        X, _ = self.split_current_pca_features_data_label()
        pca_features = [X[0:30, :], X[30:61, :], X[61:91, :], X[91:121, :],
                        X[121:151, :], X[151:181, :], X[181:211, :], X[211:241, :]]
        dict_pca_features_data = {}
        keys = ["160", "180", "200", "220", "240", "age_10", "age_50", "stuck"]
        for index, key in enumerate(keys):
            dict_pca_features_data.update({key: pca_features[index]})
        return dict_pca_features_data[option]

    # 31、返回电流信号的svm的普通训练模型
    def return_current_svm_model(self):
        X, y = self.split_current_pca_features_data_label()
        clf = svm.SVC(kernel='linear', C=1, probability=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.9)
        clf.fit(X_train, y_train)
        return clf

    # 32、根据选项option返回电流信号的svm预测的标签(没用上)
    def return_current_predict_label_svm(self, option):
        option_pca_features_data = self.spilit_current_pca_features_data(
            option)
        clf = self.return_current_svm_model()
        y_predict = clf.predict(option_pca_features_data)
        return max(list(y_predict), key=list(
            y_predict).count)  # 返回最多的那个元素 该方法列表才有

    # 33、根据选项option返回电流信号的svm预测的概率矩阵(没用上)
    def return_current_predict_proba_svm(self, option):
        option_pca_features_data = self.spilit_current_pca_features_data(
            option)
        clf = self.return_current_svm_model()
        y_predict_proba = clf.predict_proba(option_pca_features_data)
        return y_predict_proba

    # 34、对返回预测的电流信号的svm概率矩阵求平均(没用上)
    def return_current_predict_mean_proba_svm(self, option):
        y_predict_proba = self.return_current_predict_proba_svm(option)
        y_predict_proba_mean = y_predict_proba.mean(axis=0)
        return y_predict_proba_mean

    # 35、返回只带交叉验证的电流信号的svm训练模型 用于内层循环
    def return_current_kfold_predict_proba_svm(self, kfold, option):
        option_pca_features_data = self.spilit_current_pca_features_data(
            option)  # option对应的测试数据
        X, y = self.split_current_pca_features_data_label()
        state = np.random.get_state()  # 必须先打乱原来的数据集合标签集,否则初始的有序状态会导致交叉验证预测率很低
        np.random.shuffle(X)
        np.random.set_state(state)  # 保证样本和标签以相同的规律被打乱
        np.random.shuffle(y)
        Predict_proba = []  # 用于装载kfold折的概率向量 得到概率矩阵
        kf = KFold(n_splits=kfold)
        for train_index, test_index in kf.split(X):  # 内层交叉训练kfold次
            train_X, train_y = X[train_index], y[train_index]
            clf = svm.SVC(kernel='linear', C=1, probability=True)
            clf.fit(train_X, train_y)  # 训练模型
            predict_proba = clf.predict_proba(
                option_pca_features_data)  # 每一折都会生成预测概率矩阵
            predict_proba_mean = predict_proba.mean(axis=0)  # 把概率矩阵先变成概率向量
            Predict_proba.append(predict_proba_mean)  # 重新生成概率矩阵
        # print(np.array(Predict_proba).mean(axis=0))
        return np.array(Predict_proba).mean(
            axis=0)  # 但是返回继续平均生成概率向量 用于后续训练次数进行平均计算

    # 36、带有maxiter的svm的电流信号的普通训练模型 用于 交叉验证次数=0和1时调用 返回
    def return_current_maxiter_predict_proba_svm(self, option):
        option_pca_features_data = self.spilit_current_pca_features_data(
            option)  # option对应的测试数据
        X, y = self.split_current_pca_features_data_label()
        state = np.random.get_state()  # 必须先打乱原来的数据集合标签集,否则初始的有序状态会导致交叉验证预测率很低
        np.random.shuffle(X)
        np.random.set_state(state)  # 保证样本和标签以相同的规律被打乱
        np.random.shuffle(y)
        clf = svm.SVC(kernel='linear', C=1, probability=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.9)
        clf.fit(X_train, y_train)
        predict_proba = clf.predict_proba(option_pca_features_data)
        predict_proba_mean = predict_proba.mean(axis=0)  # 把概率矩阵先变成概率向量
        return list(predict_proba_mean)

    # 37、返回带交叉验证的和训练次数的电流信号的svm训练模型 用于外层循环
    # 默认采用交叉验证法 外层循环 可以使用2种算法  取决于kfold的状态
    def return_current_maxiter_kfold_predict_proba_svm(
            self, maxiter, kfold, option):
        # 外层循环调用交叉验证
        Predict_proba = []  # 用于装载maxiter次后的概率向量 得到概率矩阵
        # 进度条函数
        counter = QTime()  # 计时器
        self.ui.progressBar.setMaximum(maxiter)
        self.ui.statusbar.clearMessage()
        self.ui.statusbar.showMessage("故障预测模型开始训练...")
        self.setWindowTitle("故障预测模型正在训练当中...")
        if kfold >= 2:  # 交叉验证次数至少大于等于2 当输入1 时如果不采取措施会默认使用十次交叉验证
            # 总的训练次数为maxiter * kfold
            for i in range(maxiter):
                if self.cancel_training_flag:  # 点了取消训练时该属性变为真 那么就取消训练
                    self.ui.statusbar.showMessage("故障预测模型训练已经取消...")
                    self.setWindowTitle("故障预测模型训练取消成功")
                    return
                self.ui.progressBar.setValue(i + 1)
                QApplication.processEvents()  # 解决训练次数多时界面卡顿的语句 可以持续刷新界面
                counter.start()
                self.ui.statusbar.showMessage(
                    "故障预测模型正在进行第" + str(i + 1) + "次训练")
                predict_kfold_proba_mean = self.return_current_kfold_predict_proba_svm(
                    kfold, option)
                Predict_proba.append(predict_kfold_proba_mean)
        elif kfold == 0 or kfold == 1:
            # 训练次数为maxiter次
            # 输入为0和1时说明不使用交叉验证法 而使用留出法 也就是train_test_spilit方法
            # 不再进入 svm的 kfold内层循环算法 由于svm的普通模型函数没有maxiter的参数 所以也不调用
            # 而是重新写一个svm训练预测概率函数 调用maxiter次即可
            Predict_maxiter_proba_mean = []
            for i in range(maxiter):
                if self.cancel_training_flag:  # 点了取消训练时该属性变为真 那么就取消训练
                    self.ui.statusbar.showMessage("故障预测模型训练已经取消...")
                    self.setWindowTitle("故障预测模型训练取消成功")
                    return
                self.ui.progressBar.setValue(i + 1)
                QApplication.processEvents()  # 解决训练次数多时界面卡顿的语句 可以持续刷新界面
                counter.start()
                self.ui.statusbar.showMessage("模型正在进行第" + str(i + 1) + "次训练")
                # 返回的是列表
                predict_maxiter_proba_mean = self.return_current_maxiter_predict_proba_svm(
                    option)
                Predict_maxiter_proba_mean.append(predict_maxiter_proba_mean)
            # 其实在此语句中已经是一行向量了 但是为了返回使用一个值仍然使用Predict_proba装载
            Predict_proba.append(
                np.array(Predict_maxiter_proba_mean).mean(
                    axis=0))
            # print(Predict_proba,type(Predict_proba)) # 在这里已经是一行向量了
        # 但是要注意的问题是 虽然被重置但是只有在点击重新训练时才被重置 这个状态可能存在风险
        # 这个风险可以在训练结束时就进行重置 避免此风险
        # 结束训练后必须将状态重新更改 为False 这条语句在点击训练时实现了被重置
        self.cancel_training_flag = False  # 但是在这多一句也没有问题
        self.setWindowTitle("故障预测模型训练完毕")
        return np.array(Predict_proba).mean(axis=0)

    # 38、返回电流信号的mlp的模型
    def return_current_mlp_model(self):
        X, y = self.split_current_pca_features_data_label()
        clf = MLPClassifier(
            activation='identity',
            solver='lbfgs',
            alpha=0.1,
            hidden_layer_sizes=(
                5,
                2),
            random_state=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.9)
        clf.fit(X_train, y_train)
        return clf

    # 39、根据选项option返回电流信号的mlp预测的标签(没用上)
    def return_current_predict_label_mlp(self, option):
        option_pca_features_data = self.spilit_current_pca_features_data(
            option)
        clf = self.return_current_mlp_model()
        y_predict = clf.predict(option_pca_features_data)
        return max(list(y_predict), key=list(y_predict).count)

    # 40、根据选项option返回电流信号的预测的mlp概率矩阵
    def return_current_predict_proba_mlp(self, option):
        option_pca_features_data = self.spilit_current_pca_features_data(
            option)  # 得到指定option的pca数据
        clf = self.return_current_mlp_model()
        y_predict_proba = clf.predict_proba(option_pca_features_data)
        return y_predict_proba

    # 41、对返回预测的电流信号的mlp概率矩阵求平均
    def return_current_predict_mean_proba_mlp(self, option):
        y_predict_proba = self.return_current_predict_proba_mlp(option)
        y_predict_proba_mean = y_predict_proba.mean(axis=0)
        return y_predict_proba_mean

    # 42、返回只带交叉验证的电流信号的mlp训练模型 用于内层循环
    def return_current_kfold_predict_proba_mlp(self, kfold, option):
        option_pca_features_data = self.spilit_current_pca_features_data(
            option)  # option对应的测试数据
        X, y = self.split_current_pca_features_data_label()
        state = np.random.get_state()  # 必须先打乱原来的数据集合标签集,否则初始的有序状态会导致交叉验证预测率很低
        np.random.shuffle(X)
        np.random.set_state(state)  # 保证样本和标签以相同的规律被打乱
        np.random.shuffle(y)
        Predict_proba = []  # 用于装载kfold折的概率向量 得到概率矩阵
        kf = KFold(n_splits=kfold)
        for train_index, test_index in kf.split(X):  # 内层交叉训练kfold次
            train_X, train_y = X[train_index], y[train_index]
            clf = MLPClassifier(
                activation='identity',
                solver='lbfgs',
                alpha=0.1,
                hidden_layer_sizes=(
                    5,
                    2),
                random_state=1)
            clf.fit(train_X, train_y)  # 训练模型
            predict_proba = clf.predict_proba(
                option_pca_features_data)  # 每一折都会生成预测概率矩阵
            predict_proba_mean = predict_proba.mean(axis=0)  # 把概率矩阵先变成概率向量
            Predict_proba.append(predict_proba_mean)  # 重新生成概率矩阵
        return np.array(Predict_proba).mean(
            axis=0)  # 但是返回继续平均生成概率向量 用于后续训练次数进行平均计算

    # 43、带有maxiter的mlp的电流信号的普通训练模型 用于 交叉验证次数=0和1时调用 返回
    def return_current_maxiter_predict_proba_mlp(self, option):
        option_pca_features_data = self.spilit_current_pca_features_data(
            option)  # option对应的测试数据
        X, y = self.split_current_pca_features_data_label()
        state = np.random.get_state()  # 必须先打乱原来的数据集合标签集,否则初始的有序状态会导致交叉验证预测率很低
        np.random.shuffle(X)
        np.random.set_state(state)  # 保证样本和标签以相同的规律被打乱
        np.random.shuffle(y)
        # Predict_proba = []  # 用于装载留出法 得到的概率矩阵
        clf = MLPClassifier(
            activation='identity',
            solver='lbfgs',
            alpha=0.1,
            hidden_layer_sizes=(
                5,
                2),
            random_state=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.9)
        clf.fit(X_train, y_train)
        predict_proba = clf.predict_proba(option_pca_features_data)
        predict_proba_mean = predict_proba.mean(axis=0)  # 把概率矩阵先变成概率向量
        # Predict_proba.append(predict_proba_mean)  # 重新生成概率矩阵
        return list(predict_proba_mean)  # 列表

    # 44、返回带交叉验证的和训练次数的电流信号的mlp训练模型 用于外层循环
    def return_current_maxiter_kfold_predict_proba_mlp(
            self, maxiter, kfold, option):
        # 外层循环调用交叉验证
        Predict_proba = []  # 用于装载maxiter次后的概率向量 得到概率矩阵
        # 进度条函数
        counter = QTime()  # 计时器
        self.ui.progressBar.setMaximum(maxiter)
        self.ui.statusbar.clearMessage()
        self.ui.statusbar.showMessage("故障预测模型开始训练...")
        self.setWindowTitle("故障预测模型正在训练当中...")
        if kfold >= 2:  # 交叉验证次数至少大于等于2 当输入1 时如果不采取措施会默认使用十次交叉验证
            # 总的训练次数为maxiter * kfold
            for i in range(maxiter):
                if self.cancel_training_flag:  # 每次训练都要检测该标记(唯一会改变该标记的为取消训练按钮)
                    self.ui.statusbar.showMessage("故障预测模型训练已经取消...")
                    self.setWindowTitle("故障预测模型训练取消成功")
                    return
                self.ui.progressBar.setValue(i + 1)
                QApplication.processEvents()  # 解决训练次数多时界面卡顿的语句 可以持续刷新界面
                counter.start()
                self.ui.statusbar.showMessage("模型正在进行第" + str(i + 1) + "次训练")
                predict_kfold_proba_mean = self.return_current_kfold_predict_proba_mlp(
                    kfold, option)
                Predict_proba.append(predict_kfold_proba_mean)
            # 点击了取消按钮self.cancel_training_flag 变为了True
        elif kfold == 0 or kfold == 1:
            # 训练次数为maxiter次
            # 输入为0和1时说明不使用交叉验证法 而使用留出法 也就是train_test_spilit方法
            # 不再进入 svm的 kfold内层循环算法 由于svm的普通模型函数没有maxiter的参数 所以也不调用
            # 而是重新写一个svm训练预测概率函数 调用maxiter次即可
            Predict_maxiter_proba_mean = []
            for i in range(maxiter):
                if self.cancel_training_flag:  # 点了取消训练时该属性变为真 那么就取消训练
                    self.ui.statusbar.showMessage("故障预测模型训练已经取消...")
                    self.setWindowTitle("故障预测模型训练取消成功")
                    return
                self.ui.progressBar.setValue(i + 1)
                QApplication.processEvents()  # 解决训练次数多时界面卡顿的语句 可以持续刷新界面
                counter.start()
                self.ui.statusbar.showMessage("模型正在进行第" + str(i + 1) + "次训练")
                # 返回的是列表
                predict_maxiter_proba_mean = self.return_current_maxiter_predict_proba_mlp(
                    option)
                Predict_maxiter_proba_mean.append(predict_maxiter_proba_mean)
            # 其实在此语句中已经是一行向量了 但是为了返回使用一个值仍然使用Predict_proba装载
            Predict_proba.append(
                np.array(Predict_maxiter_proba_mean).mean(
                    axis=0))
            # print(Predict_proba,type(Predict_proba)) # 在这里已经是一行向量了
        self.setWindowTitle("故障预测模型训练完毕")
        # 对留出法相当于是[].append[1×8]
        # 但是要注意的问题是 虽然被重置但是只有在点击重新训练时才被重置 这个状态可能存在风险
        # 这个风险可以在训练结束时就进行重置 避免此风险
        # 结束训练后必须将状态重新更改 为False 这条语句在点击训练时实现了 被重置
        self.cancel_training_flag = False  # 但是在这多一句也没有问题
        return np.array(Predict_proba).mean(axis=0)

    # 45、计算带有maxiter和kfold的电流信号的模型训练耗时
    def return_current_model_traingtime(self, model, maxiter, kfold, option):
        if model == "svm":
            starttime = time.time()
            # 这里8×1
            svm = self.return_current_maxiter_kfold_predict_proba_svm(
                maxiter, kfold, option)
            endtime = time.time()
            return svm, str(round(endtime - starttime, 5))
        elif model == "mlp":
            starttime = time.time()
            mlp = self.return_current_maxiter_kfold_predict_proba_mlp(
                maxiter, kfold, option)
            # print(mlp) # 这里8×1
            endtime = time.time()
            return mlp, str(round(endtime - starttime, 5))
        else:
            print("返回模型训练时间错误...")
            return False

    # 46、读取振动波形数据的函数
    def read_vibration_waveform_data(self):
        path = os.getcwd() + "/MyPlatform/vibration_waveform_data.csv"
        waveform_data = pd.read_csv(path, encoding='gbk')
        return waveform_data

    # 47、拆分振动波形数据得到8种波形的函数
    def spilit_vibration_waveform_data(self,option):
        ''''''
        # "正常合闸", "正常分闸", "虚假合闸", "分闸不彻底",
        #                  "储能弹簧卡涩","操作机构卡涩","合闸螺栓松动", "分闸螺栓松动"
        waveform_data = self.read_vibration_waveform_data()
        keys = ["正常合闸","正常分闸", "虚假合闸", "分闸不彻底",
                "储能弹簧卡涩","操作机构卡涩","合闸螺栓松动", "分闸螺栓松动"]
        # 将所有故障波形存放在字典中，并通过option返回指定的波形
        dict_waveform_data = {}
        for index, column in enumerate(waveform_data.columns):
            dict_waveform_data.update({keys[index]: waveform_data[column]})
        option_waveform = dict_waveform_data[option]
        option_waveform.dropna(inplace=True)
        return option_waveform

    # 48、读取振动波形的pca数据
    def read_vibration_waveform_pca_features(self):
        path = os.getcwd() + "/MyPlatform/vibration_pca_features_data.csv"
        pca_features_data = pd.read_csv(path, encoding="gbk")
        return pca_features_data  # 241×2

    # 49、分离振动波形pca特征的数据和标签
    def split_vibration_pca_features_data_label(self):
        pca_features_data = self.read_vibration_waveform_pca_features()
        X_dataframe = pca_features_data.iloc[:, 0:-1]  # 分出数据和标签 此时是DataFrame格式
        y_dataframe = pca_features_data.iloc[:, -1]  # 第3列是label
        X = X_dataframe.values  # ndarray格式 样本数×维数
        y_category = y_dataframe.values  # ndarray格式
        Label = LabelEncoder()  # 初始化1个独热编码类
        y = Label.fit_transform(y_category)  # 自动生成标签 不一定按顺序标 本文的数据是标签4和7互换
        return X, y

    # 50、根据选项option得到对应的振动波形
    def spilit_vibration_pca_features_data(self,option):
        # pca的数据分离顺序是符合keys的顺序
        X, _ = self.split_vibration_pca_features_data_label()
        pca_features = [X[0:11, :], X[11:21, :], X[21:32, :], X[32:43, :],
                        X[43:54, :], X[54:64, :], X[64:73, :], X[73:84, :]]
        dict_pca_features_data = {}
        keys = ["正常合闸", "正常分闸", "分闸不彻底", "虚假合闸", "储能弹簧卡涩",
                "操作机构卡涩", "合闸螺栓松动", "分闸螺栓松动"]
        for index, key in enumerate(keys):
            dict_pca_features_data.update({key: pca_features[index]})
        return dict_pca_features_data[option]

    # 51、根据振动pca数据和label得到svm模型
    def return_vibration_svm_model(self):
        X, y = self.split_vibration_pca_features_data_label()
        clf = svm.SVC(kernel='linear', C=1, probability=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.9)
        clf.fit(X_train, y_train)
        return clf

    # 52、根据返回的振动svm模型预测其标签(用不上)
    def return_vibration_predict_label_svm(self, option):
        option_pca_features_data = self.spilit_vibration_pca_features_data(
            option)
        clf = self.return_vibration_svm_model()
        y_predict = clf.predict(option_pca_features_data)
        return max(list(y_predict), key=list(
            y_predict).count)  # 返回最多的那个元素 该方法列表才有

    # 53、根据选项option返回振动信号的svm预测的概率矩阵(用不上)
    def return_vibration_predict_proba_svm(self,option):
        option_pca_features_data = self.spilit_vibration_pca_features_data(
            option)
        clf = self.return_vibration_svm_model()
        y_predict_proba = clf.predict_proba(option_pca_features_data)
        return y_predict_proba

    # 54、对返回预测的振动信号的svm概率矩阵求平均(用不上)
    def return_vibration_predict_mean_proba_svm(self,option):
        y_predict_proba = self.return_vibration_predict_proba_svm(option)
        y_predict_proba_mean = y_predict_proba.mean(axis=0)
        return y_predict_proba_mean

    # 55、返回只带交叉验证的振动信号的svm训练模型 用于内层循环
    def return_vibration_kfold_predict_proba_svm(self,kfold, option):
        option_pca_features_data = self.spilit_vibration_pca_features_data(
            option)  # option对应的测试数据
        X, y = self.split_vibration_pca_features_data_label()
        state = np.random.get_state()  # 必须先打乱原来的数据集合标签集,否则初始的有序状态会导致交叉验证预测率很低
        np.random.shuffle(X)
        np.random.set_state(state)  # 保证样本和标签以相同的规律被打乱
        np.random.shuffle(y)
        #print(y)
        Predict_proba = []  # 用于装载kfold折的概率向量 得到概率矩阵
        kf = KFold(n_splits=kfold)
        for train_index, test_index in kf.split(X):  # 内层交叉训练kfold次
            train_X, train_y = X[train_index], y[train_index]
            clf = svm.SVC(kernel='linear', C=1, probability=True)
            clf.fit(train_X, train_y)  # 训练模型
            predict_proba = clf.predict_proba(
                option_pca_features_data)  # 每一折都会生成预测概率矩阵
            predict_proba_mean = predict_proba.mean(axis=0)  # 把概率矩阵先变成概率向量
            Predict_proba.append(predict_proba_mean)  # 重新生成概率矩阵
        # print(np.array(Predict_proba).mean(axis=0))
        return np.array(Predict_proba).mean(
            axis=0)  # 但是返回继续平均生成概率向量 用于后续训练次数进行平均计算

    # 56、带有maxiter的svm的振动信号的普通训练模型 用于 交叉验证次数=0和1时调用 返回
    def return_vibration_maxiter_predict_proba_svm(self,option):
        # 返回振动信号的训练次数平均概率
        option_pca_features_data = self.spilit_vibration_pca_features_data(
            option)  # option对应的测试数据
        X, y = self.split_vibration_pca_features_data_label()
        state = np.random.get_state()  # 必须先打乱原来的数据集合标签集,否则初始的有序状态会导致交叉验证预测率很低
        np.random.shuffle(X)
        np.random.set_state(state)  # 保证样本和标签以相同的规律被打乱
        np.random.shuffle(y)
        clf = svm.SVC(kernel='linear', C=1, probability=True)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.9)
        clf.fit(X_train, y_train)
        predict_proba = clf.predict_proba(option_pca_features_data)
        predict_proba_mean = predict_proba.mean(axis=0)  # 把概率矩阵先变成概率向量
        return list(predict_proba_mean)

    # 57、返回带交叉验证的和训练次数的振动信号的svm训练模型 用于外层循环
    # 默认采用交叉验证法 外层循环 可以使用2种算法  取决于kfold的状态
    def return_vibration_maxiter_kfold_predict_proba_svm(
            self, maxiter, kfold, option):
        # 外层循环调用交叉验证
        Predict_proba = []  # 用于装载maxiter次后的概率向量 得到概率矩阵
        # 进度条函数
        counter = QTime()  # 计时器
        self.ui.progressBar.setMaximum(maxiter)
        self.ui.statusbar.clearMessage()
        self.ui.statusbar.showMessage("故障预测模型开始训练...")
        self.setWindowTitle("故障预测模型正在训练当中...")
        if kfold >= 2:  # 交叉验证次数至少大于等于2 当输入1 时如果不采取措施会默认使用十次交叉验证
            # 总的训练次数为maxiter * kfold
            for i in range(maxiter):
                if self.cancel_training_flag:  # 点了取消训练时该属性变为真 那么就取消训练
                    self.ui.statusbar.showMessage("故障预测模型训练已经取消...")
                    self.setWindowTitle("故障预测模型训练取消成功")
                    return
                self.ui.progressBar.setValue(i + 1)
                QApplication.processEvents()  # 解决训练次数多时界面卡顿的语句 可以持续刷新界面
                counter.start()
                self.ui.statusbar.showMessage(
                    "故障预测模型正在进行第" + str(i + 1) + "次训练")
                predict_kfold_proba_mean = self.return_vibration_kfold_predict_proba_svm(
                    kfold, option)
                Predict_proba.append(predict_kfold_proba_mean)
        elif kfold == 0 or kfold == 1:
            # 训练次数为maxiter次
            # 输入为0和1时说明不使用交叉验证法 而使用留出法 也就是train_test_spilit方法
            # 不再进入 svm的 kfold内层循环算法 由于svm的普通模型函数没有maxiter的参数 所以也不调用
            # 而是重新写一个svm训练预测概率函数 调用maxiter次即可
            Predict_maxiter_proba_mean = []
            for i in range(maxiter):
                if self.cancel_training_flag:  # 点了取消训练时该属性变为真 那么就取消训练
                    self.ui.statusbar.showMessage("故障预测模型训练已经取消...")
                    self.setWindowTitle("故障预测模型训练取消成功")
                    return
                self.ui.progressBar.setValue(i + 1)
                QApplication.processEvents()  # 解决训练次数多时界面卡顿的语句 可以持续刷新界面
                counter.start()
                self.ui.statusbar.showMessage("模型正在进行第" + str(i + 1) + "次训练")
                # 返回的是列表
                predict_maxiter_proba_mean = self.return_vibration_maxiter_predict_proba_svm(
                    option)
                Predict_maxiter_proba_mean.append(predict_maxiter_proba_mean)
            # 其实在此语句中已经是一行向量了 但是为了返回使用一个值仍然使用Predict_proba装载
            Predict_proba.append(
                np.array(Predict_maxiter_proba_mean).mean(
                    axis=0))
            # print(Predict_proba,type(Predict_proba)) # 在这里已经是一行向量了
        # 但是要注意的问题是 虽然被重置但是只有在点击重新训练时才被重置 这个状态可能存在风险
        # 这个风险可以在训练结束时就进行重置 避免此风险
        # 结束训练后必须将状态重新更改 为False 这条语句在点击训练时实现了被重置
        self.cancel_training_flag = False  # 但是在这多一句也没有问题
        self.setWindowTitle("故障预测模型训练完毕")
        return np.array(Predict_proba).mean(axis=0)

    # 58、返回振动信号的mlp的模型
    def return_vibration_mlp_model(self):
        X, y = self.split_vibration_pca_features_data_label()
        clf = MLPClassifier(
            activation='identity',
            solver='lbfgs',
            alpha=0.1,
            hidden_layer_sizes=(
                5,
                2),
            random_state=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.9)
        clf.fit(X_train, y_train)
        return clf

    # 59、根据选项option返回振动信号的mlp预测的标签(用不上)
    def return_vibration_predict_label_mlp(self,option):
        option_pca_features_data = self.spilit_vibration_pca_features_data(
            option)
        clf = self.return_vibration_mlp_model()
        y_predict = clf.predict(option_pca_features_data)
        return max(list(y_predict), key=list(y_predict).count)

    # 60、根据选项option返回振动信号的预测的mlp概率矩阵
    def return_vibration_predict_proba_mlp(self,option):
        option_pca_features_data = self.spilit_vibration_pca_features_data(
            option)  # 得到指定option的pca数据
        clf = self.return_vibration_mlp_model()
        y_predict_proba = clf.predict_proba(option_pca_features_data)
        return y_predict_proba

    # 61、对返回预测的振动信号的mlp概率矩阵求平均
    def return_vibration_predict_mean_proba_mlp(self,option):
        y_predict_proba = self.return_vibration_predict_proba_mlp(option)
        y_predict_proba_mean = y_predict_proba.mean(axis=0)
        return y_predict_proba_mean

    # 62、返回只带交叉验证的振动信号的mlp训练模型 用于内层循环
    def return_vibration_kfold_predict_proba_mlp(self,kfold, option):
        option_pca_features_data = self.spilit_vibration_pca_features_data(
            option)  # option对应的测试数据
        X, y = self.split_vibration_pca_features_data_label()
        state = np.random.get_state()  # 必须先打乱原来的数据集合标签集,否则初始的有序状态会导致交叉验证预测率很低
        np.random.shuffle(X)
        np.random.set_state(state)  # 保证样本和标签以相同的规律被打乱
        np.random.shuffle(y)
        Predict_proba = []  # 用于装载kfold折的概率向量 得到概率矩阵
        kf = KFold(n_splits=kfold)
        for train_index, test_index in kf.split(X):  # 内层交叉训练kfold次
            train_X, train_y = X[train_index], y[train_index]
            clf = MLPClassifier(
                activation='identity',
                solver='lbfgs',
                alpha=0.1,
                hidden_layer_sizes=(
                    5,
                    2),
                random_state=1)
            clf.fit(train_X, train_y)  # 训练模型
            predict_proba = clf.predict_proba(
                option_pca_features_data)  # 每一折都会生成预测概率矩阵
            predict_proba_mean = predict_proba.mean(axis=0)  # 把概率矩阵先变成概率向量
            Predict_proba.append(predict_proba_mean)  # 重新生成概率矩阵
        return np.array(Predict_proba).mean(
            axis=0)  # 但是返回继续平均生成概率向量 用于后续训练次数进行平均计算

    # 63、带有maxiter的mlp的振动信号的普通训练模型 用于 交叉验证次数=0和1时调用 返回
    def return_vibration_maxiter_predict_proba_mlp(self,option):
        option_pca_features_data = self.spilit_vibration_pca_features_data(
            option)  # option对应的测试数据
        X, y = self.split_vibration_pca_features_data_label()
        state = np.random.get_state()  # 必须先打乱原来的数据集合标签集,否则初始的有序状态会导致交叉验证预测率很低
        np.random.shuffle(X)
        np.random.set_state(state)  # 保证样本和标签以相同的规律被打乱
        np.random.shuffle(y)
        # Predict_proba = []  # 用于装载留出法 得到的概率矩阵
        clf = MLPClassifier(
            activation='identity',
            solver='lbfgs',
            alpha=0.1,
            hidden_layer_sizes=(
                5,
                2),
            random_state=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.9)
        clf.fit(X_train, y_train)
        predict_proba = clf.predict_proba(option_pca_features_data)
        predict_proba_mean = predict_proba.mean(axis=0)  # 把概率矩阵先变成概率向量
        # Predict_proba.append(predict_proba_mean)  # 重新生成概率矩阵
        return list(predict_proba_mean)  # 列表

    # 64、返回带交叉验证的和训练次数的振动信号的mlp训练模型 用于外层循环
    def return_vibration_maxiter_kfold_predict_proba_mlp(
            self, maxiter, kfold, option):
        # 外层循环调用交叉验证
        Predict_proba = []  # 用于装载maxiter次后的概率向量 得到概率矩阵
        # 进度条函数
        counter = QTime()  # 计时器
        self.ui.progressBar.setMaximum(maxiter)
        self.ui.statusbar.clearMessage()
        self.ui.statusbar.showMessage("故障预测模型开始训练...")
        self.setWindowTitle("故障预测模型正在训练当中...")
        if kfold >= 2:  # 交叉验证次数至少大于等于2 当输入1 时如果不采取措施会默认使用十次交叉验证
            # 总的训练次数为maxiter * kfold
            for i in range(maxiter):
                if self.cancel_training_flag:  # 每次训练都要检测该标记(唯一会改变该标记的为取消训练按钮)
                    self.ui.statusbar.showMessage("故障预测模型训练已经取消...")
                    self.setWindowTitle("故障预测模型训练取消成功")
                    return
                self.ui.progressBar.setValue(i + 1)
                QApplication.processEvents()  # 解决训练次数多时界面卡顿的语句 可以持续刷新界面
                counter.start()
                self.ui.statusbar.showMessage("模型正在进行第" + str(i + 1) + "次训练")
                predict_kfold_proba_mean = self.return_vibration_kfold_predict_proba_mlp(
                    kfold, option)
                Predict_proba.append(predict_kfold_proba_mean)
            # 点击了取消按钮self.cancel_training_flag 变为了True
        elif kfold == 0 or kfold == 1:
            # 训练次数为maxiter次
            # 输入为0和1时说明不使用交叉验证法 而使用留出法 也就是train_test_spilit方法
            # 不再进入 svm的 kfold内层循环算法 由于svm的普通模型函数没有maxiter的参数 所以也不调用
            # 而是重新写一个svm训练预测概率函数 调用maxiter次即可
            Predict_maxiter_proba_mean = []
            for i in range(maxiter):
                if self.cancel_training_flag:  # 点了取消训练时该属性变为真 那么就取消训练
                    self.ui.statusbar.showMessage("故障预测模型训练已经取消...")
                    self.setWindowTitle("故障预测模型训练取消成功")
                    return
                self.ui.progressBar.setValue(i + 1)
                QApplication.processEvents()  # 解决训练次数多时界面卡顿的语句 可以持续刷新界面
                counter.start()
                self.ui.statusbar.showMessage("模型正在进行第" + str(i + 1) + "次训练")
                # 返回的是列表
                predict_maxiter_proba_mean = self.return_vibration_maxiter_predict_proba_mlp(
                    option)
                Predict_maxiter_proba_mean.append(predict_maxiter_proba_mean)
            # 其实在此语句中已经是一行向量了 但是为了返回使用一个值仍然使用Predict_proba装载
            Predict_proba.append(
                np.array(Predict_maxiter_proba_mean).mean(
                    axis=0))
            # print(Predict_proba,type(Predict_proba)) # 在这里已经是一行向量了
        self.setWindowTitle("故障预测模型训练完毕")
        # 对留出法相当于是[].append[1×8]
        # 但是要注意的问题是 虽然被重置但是只有在点击重新训练时才被重置 这个状态可能存在风险
        # 这个风险可以在训练结束时就进行重置 避免此风险
        # 结束训练后必须将状态重新更改 为False 这条语句在点击训练时实现了 被重置
        self.cancel_training_flag = False  # 但是在这多一句也没有问题
        return np.array(Predict_proba).mean(axis=0)

    # 65、计算带有maxiter和kfold的电流信号的模型训练耗时
    def return_vibration_model_traingtime(self, model, maxiter, kfold, option):
        if model == "svm":
            starttime = time.time()
            # 这里8×1
            svm = self.return_vibration_maxiter_kfold_predict_proba_svm(
                maxiter, kfold, option)
            endtime = time.time()
            return svm, str(round(endtime - starttime, 5))
        elif model == "mlp":
            starttime = time.time()
            mlp = self.return_vibration_maxiter_kfold_predict_proba_mlp(
                maxiter, kfold, option)
            # print(mlp) # 这里8×1
            endtime = time.time()
            return mlp, str(round(endtime - starttime, 5))
        else:
            print("返回模型训练时间错误...")
            return False

    '''3、后台未来可能用到的其他函数'''
    # 66、返回颜色的RGB值
    def return_cmaps(self):
        cnames = {
            'aliceblue': '#F0F8FF',
            'antiquewhite': '#FAEBD7',
            'aqua': '#00FFFF',
            'aquamarine': '#7FFFD4',
            'azure': '#F0FFFF',
            'beige': '#F5F5DC',
            'bisque': '#FFE4C4',
            'black': '#000000',
            'blanchedalmond': '#FFEBCD',
            'blue': '#0000FF',
            'blueviolet': '#8A2BE2',
            'brown': '#A52A2A',
            'burlywood': '#DEB887',
            'cadetblue': '#5F9EA0',
            'chartreuse': '#7FFF00',
            'chocolate': '#D2691E',
            'coral': '#FF7F50',
            'cornflowerblue': '#6495ED',
            'cornsilk': '#FFF8DC',
            'crimson': '#DC143C',
            'cyan': '#00FFFF',
            'darkblue': '#00008B',
            'darkcyan': '#008B8B',
            'darkgoldenrod': '#B8860B',
            'darkgray': '#A9A9A9',
            'darkgreen': '#006400',
            'darkkhaki': '#BDB76B',
            'darkmagenta': '#8B008B',
            'darkolivegreen': '#556B2F',
            'darkorange': '#FF8C00',
            'darkorchid': '#9932CC',
            'darkred': '#8B0000',
            'darksalmon': '#E9967A',
            'darkseagreen': '#8FBC8F',
            'darkslateblue': '#483D8B',
            'darkslategray': '#2F4F4F',
            'darkturquoise': '#00CED1',
            'darkviolet': '#9400D3',
            'deeppink': '#FF1493',
            'deepskyblue': '#00BFFF',
            'dimgray': '#696969',
            'dodgerblue': '#1E90FF',
            'firebrick': '#B22222',
            'floralwhite': '#FFFAF0',
            'forestgreen': '#228B22',
            'fuchsia': '#FF00FF',
            'gainsboro': '#DCDCDC',
            'ghostwhite': '#F8F8FF',
            'gold': '#FFD700',
            'goldenrod': '#DAA520',
            'gray': '#808080',
            'green': '#008000',
            'greenyellow': '#ADFF2F',
            'honeydew': '#F0FFF0',
            'hotpink': '#FF69B4',
            'indianred': '#CD5C5C',
            'indigo': '#4B0082',
            'ivory': '#FFFFF0',
            'khaki': '#F0E68C',
            'lavender': '#E6E6FA',
            'lavenderblush': '#FFF0F5',
            'lawngreen': '#7CFC00',
            'lemonchiffon': '#FFFACD',
            'lightblue': '#ADD8E6',
            'lightcoral': '#F08080',
            'lightcyan': '#E0FFFF',
            'lightgoldenrodyellow': '#FAFAD2',
            'lightgreen': '#90EE90',
            'lightgray': '#D3D3D3',
            'lightpink': '#FFB6C1',
            'lightsalmon': '#FFA07A',
            'lightseagreen': '#20B2AA',
            'lightskyblue': '#87CEFA',
            'lightslategray': '#778899',
            'lightsteelblue': '#B0C4DE',
            'lightyellow': '#FFFFE0',
            'lime': '#00FF00',
            'limegreen': '#32CD32',
            'linen': '#FAF0E6',
            'magenta': '#FF00FF',
            'maroon': '#800000',
            'mediumaquamarine': '#66CDAA',
            'mediumblue': '#0000CD',
            'mediumorchid': '#BA55D3',
            'mediumpurple': '#9370DB',
            'mediumseagreen': '#3CB371',
            'mediumslateblue': '#7B68EE',
            'mediumspringgreen': '#00FA9A',
            'mediumturquoise': '#48D1CC',
            'mediumvioletred': '#C71585',
            'midnightblue': '#191970',
            'mintcream': '#F5FFFA',
            'mistyrose': '#FFE4E1',
            'moccasin': '#FFE4B5',
            'navajowhite': '#FFDEAD',
            'navy': '#000080',
            'oldlace': '#FDF5E6',
            'olive': '#808000',
            'olivedrab': '#6B8E23',
            'orange': '#FFA500',
            'orangered': '#FF4500',
            'orchid': '#DA70D6',
            'palegoldenrod': '#EEE8AA',
            'palegreen': '#98FB98',
            'paleturquoise': '#AFEEEE',
            'palevioletred': '#DB7093',
            'papayawhip': '#FFEFD5',
            'peachpuff': '#FFDAB9',
            'peru': '#CD853F',
            'pink': '#FFC0CB',
            'plum': '#DDA0DD',
            'powderblue': '#B0E0E6',
            'purple': '#800080',
            'red': '#FF0000',
            'rosybrown': '#BC8F8F',
            'royalblue': '#4169E1',
            'saddlebrown': '#8B4513',
            'salmon': '#FA8072',
            'sandybrown': '#FAA460',
            'seagreen': '#2E8B57',
            'seashell': '#FFF5EE',
            'sienna': '#A0522D',
            'silver': '#C0C0C0',
            'skyblue': '#87CEEB',
            'slateblue': '#6A5ACD',
            'slategray': '#708090',
            'snow': '#FFFAFA',
            'springgreen': '#00FF7F',
            'steelblue': '#4682B4',
            'tan': '#D2B48C',
            'teal': '#008080',
            'thistle': '#D8BFD8',
            'tomato': '#FF6347',
            'turquoise': '#40E0D0',
            'violet': '#EE82EE',
            'wheat': '#F5DEB3',
            'white': '#FFFFFF',
            'whitesmoke': '#F5F5F5',
            'yellow': '#FFFF00',
            'yellowgreen': '#9ACD32'}
        return cnames

    # 67、绘图函数
    def plot(self,x,y,legendname=None,titlename=None,xlabelname=None,ylabelname=None,Color='k'):
        '''
        :param x:  自变量
        :param y: 因变量
        :param legendname: 图例
        :param titlename: 标题
        :param xlabelname: 轴标签
        :param ylabelname: 轴标签
        :param Color: 颜色，如果不指定颜色 则随机选择颜色绘图
        :return: 图像
        '''
        '''设置图片大小，分辨率，[图像编号]，背景色，是否显示边框 ， 边框颜色'''
        '''线条类型
        '-'  |实线       '--'|虚线
        '-.  |虚点线     ':'|点线
        '.'  |点        ','|像素点
        'o'  |圆点      'v'|下三角点
        '^'  |上三角点   '<'|左三角点
        '>'  |右三角点   '1'|下三叉点
        '2'  |上三叉点   '3'|左三叉点
        '4'  |右三叉点   's'|正方点
        'p'  |五角点     '*'|星形点
        'h'  |六边形点1  'H'|六边形点2
        '+'  |加号点     'x'|乘号点
        'D'  |实心菱形点  'd'|瘦菱形点
        '_'  |横线点
        '''
        cnames = self.return_cmaps()  # 得到color得16进制值
        if Color:
            color = Color
        else:
            color = random.sample(list(cnames.values()), 2)
        '''颜色类型 ：bgrcmykw ：蓝绿红青、品红、黄黑白'''
        plt.figure(figsize=(8,4),dpi=144,facecolor='w',frameon=True,edgecolor='b')

        '''设置图例文字，线颜色，线型，线宽，线风格，标记类型，标记尺寸，标记颜色，透明度'''
        plt.plot(x,y,label=legendname,color=color[0],linewidth=1.5,linestyle='--',
            marker='o',markersize='2',markerfacecolor=color[0],alpha=0.8)

        '''设置图例的文字，位置 ， 文字大小， 多图例按列数显示(默认按行)， 图例标记为原尺寸的多少倍大小 ， 是否添加阴影 ， 图里圆角方角，是否带边框， 边框透明度'''
        plt.legend(
            (legendname,
             ),
            loc='upper right',
            fontsize=18,
            ncol=1,
            markerscale=1,
            shadow=False,
            fancybox=True,
            frameon=True,
            framealpha=0.5)

        '''设置网格线是否出现、显示哪个轴的网格线 ，网格线颜色， 风格、 宽度、 设置次刻度线（default = major）'''
        plt.grid(
            b=None,
            axis='x',
            color='c',
            linestyle='-',
            linewidth=8,
            which='major')

        '''设置坐标轴范围'''
        ymax = max(y)
        ymin = min(y)
        xmax = max(x)
        xmin = min(x)
        plt.ylim(ymin, ymax)
        plt.xlim(xmin, xmax)

        '''设置轴标签是否旋转，文字垂直和水平位置'''
        plt.xlabel(xlabelname, fontsize=18, rotation=None)
        plt.ylabel(ylabelname, fontsize=18, rotation=90)

        '''设置轴刻度'''
        plt.yticks(())
        plt.xticks(())

        '''设置哪个轴的属性、次刻度线设置， 刻度线显示方式(绘图区内测、外侧、同时显示)，刻度线宽度，刻度线颜色、刻度线标签颜色(任命/日期等)
        刻度线与标签距离、刻度线标签大小、刻度线是否显示(default=bottom ,left)、刻度线标签是否显示(default = bottom/left)'''
        plt.tick_params(
            axis='both',
            which='major',
            direction='in',
            width=1,
            length=3,
            color='k',
            labelcolor='k',
            pad=1,
            labelsize=15,
            bottom=True,
            right=True,
            top=False,
            labeltop=False,
            labelbottom=True,
            labelleft=True,
            labelright=False)

        '''方框外形、背景色、 透明度、方框粗细、方框到文字的距离'''
        bb = dict(boxstyle='round', fc='w', ec='m', alpha=0.8, lw=10, pad=1.2)

        '''设置标题文字大小、标题大小、标题正常/斜体/倾斜、垂直位置、水平位置、透明度、标题背景色、是否旋转、标题边框有关设置（字典格式）'''
        plt.title(
            label=titlename,
            fontsize=20,
            fontweight='normal',
            fontstyle='italic',
            alpha=0.8,
            backgroundcolor='w',
            rotation=None)  # bbox = bb

        plt.show()

    # 68、数据归一化函数
    def data_normalization(self,array, mode=0, boundary=0, reverse=False, col_max=None, col_min=None, row_max=None,
                         row_min=None):
        '''
        :param array: 要归一化或者反归一化的数组 输入为数组格式
        :param mode: mode=0表示按行归一化,mode=1表示按列归一化
        按行归一化,需要得到每列的最值,然后某一列的每行数据都用这个最大最小值归一化 那么不同行之间分布在[0,1]或[-1,1]之间
        :param reverse: bool 反归一化
        :param col_max: 按行反归一化时需要原始数据每列的最大值 行向量
        :param col_min: 按行反归一化时需要原始数据每列的最小值 行向量
        :param row_max: 按列反归一化时需要原始数据每行的最大值 行向量
        :param row_min: 按列反归一化时需要原始数据每行的最小值 行向量
        :return: 归一化或反归一化后的数组
        测试程序 :
        array =[[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20] ]
        array = np.array(array)
        norm_array = MaxMinNormalized(array,mode=1,boundary=0)
        # 按行归一化 需要提供每行的最大值和最小值 使用axis=1
        new_array = MaxMinNormalized(norm_array,mode=1,boundary=0,reverse=True,row_max=array.max(axis=1),row_min=array.min(axis=1))
        '''
        data_shape = array.shape  # 返回数组的行列数
        data_rows = data_shape[0]  # 行数
        data_cols = data_shape[1]  # 列数
        # print(data_rows,data_cols)
        norm = np.empty((data_rows, data_cols))  # 用于存放归一化后的数据
        reverse_norm = np.empty((data_rows, data_cols))  # 用于存放反归一化后的数据
        if not reverse:
            if mode == 0:  # 选择按行归一化模式 即只有一行为-1或者1
                maxcols = array.max(axis=0)  # 返回每列的最大值，生成的是行向量
                mincols = array.min(axis=0)  # 返回每列的最小值，生成的是行向量
                # print(maxcols,mincols)
                # print(maxcols, mincols)
                for i in range(data_cols):  # 循环对每行依次处理
                    # 按行归一化是对每列操作
                    # 每列应用公式 (x - xmin) / (xmax - xmin),[-1,1]之间则是2*(x - xmin) / (xmax - xmin)-1
                    # 每一列的元素都减去该列的最小值 再除(该列最大值-该列最小值)
                    if boundary == -1:  # 归一化到[-1,1]
                        norm[:, i] = (2 * (array[:, i] - mincols[i]) / (maxcols[i] - mincols[i])) - 1
                    elif boundary == 0:  # 归一化到[0,1]
                        norm[:, i] = (array[:, i] - mincols[i]) / (maxcols[i] - mincols[i])
                    else:
                        print("boundary只能选择0或-1,对应归一化范围是[0,1]或[-1,1]")
                        return False
                return norm
            if mode == 1:  # 选择按列归一化模式 即只有一列为-1或者1
                maxrows = array.max(axis=1)  # 返回每行的最大值
                minrows = array.min(axis=1)  # 返回每行的最小值
                # print(maxrows,minrows)
                for i in range(data_rows):  # 循环对每行依次处理
                    if boundary == -1:
                        # 每行元素依次减去行的最大值 再除(该行最大值-该行最小值)
                        norm[i, :] = (2 * (array[i, :] - minrows[i]) / (maxrows[i] - minrows[i])) - 1
                    elif boundary == 0:
                        norm[i, :] = (array[i, :] - minrows[i]) / (maxrows[i] - minrows[i])
                    else:
                        print("boundary只能选择0或-1,对应归一化范围是[0,1]或[-1,1]")
                        return False
                return norm
        elif reverse:  # reverse 为True执行下述操作
            if mode == 0:
                maxcols = array.max(axis=0)
                mincols = array.min(axis=0)
                for i in range(data_cols):
                    if boundary == -1:  # 按[-1,1]反归一化
                        # (x_norm + 1) / 2 * (xmax - xmin) + xmin
                        # 原数组每列的最大值和最小值 x_norm *(xmax-xmin)+xmin
                        reverse_norm[:, i] = ((array[:, i] + 1) / 2) * (col_max[i] - col_min[i]) + col_min[i]
                    elif boundary == 0:  # 按[0,1]反归一化
                        # 或 x_norm * (xmax - xmin) + xmin
                        reverse_norm[:, i] = array[:, i] * (col_max[i] - col_min[i]) + col_min[i]
                    else:
                        print("boundary只能选择0或-1,对应归一化范围是[0,1]或[-1,1]")
                        return False
                return reverse_norm
            if mode == 1:
                maxrows = array.max(axis=1)
                minrows = array.min(axis=1)
                for i in range(data_rows):
                    if boundary == -1:
                        reverse_norm[i, :] = ((array[i, :] + 1) / 2) * \
                                             (row_max[i] - row_min[i]) + row_min[i]
                    elif boundary == 0:
                        reverse_norm[i, :] = array[i, :] * (row_max[i] - row_min[i]) + row_min[i]
                    else:
                        print("boundary只能选择0或-1,对应归一化范围是[0,1]或[-1,1]")
                        return False
                return reverse_norm

    '''4、振动信号处理用到函数'''
    '''4.1、数据处理函数'''
    # 69、读取单一txt数据的函数--->得到Series数据
    def read_txtdata_to_series(self,path, num):
        '''
        :param path: txt文件路径
        :param num:  文件名的数字编号 一般调用此函数前先将文件夹下所有txt文件名改为顺序数字
        :return: series
        '''
        alm = []
        with open(path + "\\" + str(num) + ".txt", encoding='utf-8') as f:
            for i, line in enumerate(f.readlines()):
                alm.append(float(line))  # float 防止读取的是字符和换行符
            f.close()
        alm_series = pd.Series(alm)
        return alm_series

    # 70、读取文件夹下的所有txt文件 最终合并成Dataframe 10×4500的数据(1种故障类型测10组数据)
    def read_all_txtdata_to_dataframe(self,path):
        '''
        :param path: 存放所有txt文件的文件夹路径
        :return: dataframe 某个故障类型所有样本的拼接
        '''
        alm_dataframe = pd.DataFrame()
        files = os.listdir(path)  # 列出文件夹下所有的文件
        for i in range(len(files)):
            alm_series = self.read_txtdata_to_series(path, i + 1)
            # print(i+1) #检查是否30个文件都读取了
            # print(len(alm_series)) #检查数据是否都是10000长度
            # 按列拼接
            alm_dataframe = pd.concat([alm_dataframe, alm_series],
                                      axis=1, ignore_index=True)
        return alm_dataframe

    # 71、频谱分析
    def fft_data(self,X, N, title):
        # X : 原始数据 ndarray N : 采样率
        fft_y = fft(X)
        N1 = len(X)
        f1 = np.arange(N1)
        abs_y = np.abs(fft_y)
        half_f = np.arange(0, N, N / N1)
        half_f = f1[range(int(N1 / 2))]
        half_abs_y = abs_y[range(int(N1 / 2))]
        plt.plot(half_f[10:-1], half_abs_y[10:-1] / (N / 2))  # 开头几个太大了不要
        # plt.plot(half_f[10:-1], half_abs_y[10:-1] / max(half_abs_y[10:-1]) * 10 ) # 先限制到0~1 再归一化到10V
        plt.xlabel("频率/Hz")
        plt.ylabel("振幅")
        plt.title(title + "振动信号的频谱图")
        plt.show()

    # 72、EMD可视化(本论文使用的matlab绘制emd分解 但是此函数也可以使用)
    def plot_imfs(self,signal, imfs, title, time_samples=None, fignum=None):
        if time_samples is None:
            time_samples = np.arange(signal.shape[0])
        n_imfs = imfs.shape[0]
        plt.figure(num=fignum)
        axis_extent = max(np.max(np.abs(imfs[:-1, :]), axis=0))
        ax = plt.subplot(n_imfs + 1, 1, 1)
        ax.plot(time_samples, signal)
        ax.axis([time_samples[0], time_samples[-1], signal.min(), signal.max()])
        ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                       labelbottom=False)
        ax.grid(False)
        ax.set_ylabel('signal')
        ax.set_title(title)
        # Plot the IMFs
        for i in range(n_imfs - 1):
            print(i + 2)
            ax = plt.subplot(n_imfs + 1, 1, i + 2)
            ax.plot(time_samples, imfs[i, :])
            ax.axis([time_samples[0], time_samples[-1], -axis_extent, axis_extent])
            ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                           labelbottom=False)
            ax.grid(False)
            ax.set_ylabel('imf' + str(i + 1))
        # Plot the residue
        ax = plt.subplot(n_imfs + 1, 1, n_imfs + 1)
        ax.plot(time_samples, imfs[-1, :], 'r')
        ax.axis('tight')
        ax.tick_params(which='both', left=False, bottom=False, labelleft=False,
                       labelbottom=False)
        ax.grid(False)
        ax.set_ylabel('res')
        plt.show()

    # 73、读取某个故障类型下的某个emd的csv数据 1.csv : 在计算熵中调用
    def read_one_dataframe(self,path, num):
        # path =r"C:\Users\chenbei\Desktop\钢\振动信号数据\csv数据\分闸不彻底4500"
        data = pd.read_csv(path + "\\" + str(num) + ".csv")
        return data

    '''4.2、特征提取函数'''
    # 74、计算两个Series的互相关系数
    def correlation_coefficient(self,a, b):
        A = pd.Series(a)
        B = pd.Series(b)
        Corr = round(A.corr(B), 5)
        return Corr

    # 75、计算原始信号和imf分量之间的互相关系数 得到列表
    def imf_signal_correlation(self,signal, IMF):
        IMF = IMF.T  # 4500×7
        Corraltion = []
        # print(IMF.shape[0],IMF.shape[1])
        for i in range(IMF.shape[1]):
            imf = IMF[:, i]
            # print(len(imf)) # 4500
            corr = self.correlation_coefficient(imf, signal)
            # print(corr)
            Corraltion.append(abs(corr))
        return Corraltion

    # 76、计算某个序列的互相关熵
    def signal_correlation_entropy(self,corr):
        entopy = 0
        for i in range(len(corr)):
            if corr[i] == 0:
                print('存在互相关系数为0,程序将跳过该元素防止出错')
                tem = 0
            else:
                tem = corr[i] * np.log10(corr[i])
            entopy = entopy + tem
        entopy = entopy * (-1)
        return entopy

    # 77、可视化不同imf分解程度的emd能量、svd奇异值、corr相关系数
    def plot_energy_svd_corr_imf(self,IMF, energy, svd, corr):
        imfn = np.linspace(1, len(IMF), len(IMF))
        plt.plot(imfn, energy, label='能量熵')
        plt.plot(imfn, svd, label='奇异熵')
        plt.plot(imfn, corr, label='互相关熵')
        plt.xlabel('IMF分量')
        plt.ylabel('信息熵')
        plt.tight_layout()
        plt.legend()
        plt.show()

    # 78、绘制energy、svd、corr之间的关系
    def plot_energy_svd_corr(self,energy, svd, corr, title):
        imf_len = np.arange(1, len(energy) + 1, 1)  # start end step
        # linspace : start end N
        # print(imf_len)
        plt.plot(imf_len, energy, 'r-p', label='归一化能量', linewidth=1.5)
        plt.plot(imf_len, svd, 'b-o', label='归一化奇异值', linewidth=1.5)
        plt.plot(imf_len, corr, 'g-*', label='归一化相关系数', linewidth=1.5)
        plt.xlabel("imf分量")
        plt.ylabel("归一化值")
        plt.title(title)
        plt.legend()
        plt.show()

    # 79、计算某个故障类型的熵特征和时域特征 返回9× 10&11的特征矩阵 dataframe
    def calculate_all_dataframe_features(self,filepath, csvpath, title):
        '''
        :param filepath: 存放csv文件夹的路径
        :param csvpath: csv文件名
        :param title: 可视化energy、svd、corr的图标题
        :return: 某个故障类型所有样本的特征 , 所有样本svd、energy、corr的平均值
        '''
        files = os.listdir(filepath)
        source_data = pd.read_csv(csvpath)  # 10列原始数据
        Proba_energy = []
        Proba_svd = []
        Proba_corr = []
        features_dataframe = pd.DataFrame()  # 用于存放所有波形的熵和时域特征 9 × 11&10
        for i in range(len(files) - 1):  # 排除一个文件 11 -1 = 10个文件
            # 1、按顺序读取原始波形 共10组
            column_data = source_data.iloc[:, i]  # 某组原始数据 4500×1

            # 2、读取对应波形的emd数据
            emd_data = self.read_one_dataframe(filepath, i + 1)  # 该组原始数据对应的emd数据 4500×7
            emd_data = emd_data.values.T

            # 3、计算互相关系数熵
            # ①必须取值②必须转置 该函数处理的是7×4500的emd数据
            corrlation = self.imf_signal_correlation(column_data, emd_data)  # 得到每个imf分量与原始信号的相关系数
            proba_corr = corrlation[0:-1] / sum(corrlation[0:-1])  # 不要残差 这里0:-1不一定是6个 很可能是7个 但是一定是超过6个的
            # print(len(corrlation))
            # 得到当前故障类型 分合闸不彻底下每组波形的互相关系数熵
            corrEntropy = self.signal_correlation_entropy(proba_corr)
            if len(proba_corr) >= 6:
                if len(proba_corr) == 6:
                    Proba_corr.append(proba_corr)
                else:
                    Proba_corr.append(proba_corr[0:6])

            # 4、计算emd能量熵
            Energy = emd_entropy(emd_data.T)  # 实例化对象
            # energy = Energy.get_energy() # 单一样本的所有imf分量的能量
            # print(len(energy))
            proba_energy = Energy.get_pe()  # 单一样本所有imf分量的比值
            if len(proba_energy) >= 6:
                if len(proba_energy) == 6:
                    Proba_energy.append(proba_energy)
                else:
                    Proba_energy.append(proba_energy[0:6])
            # 得到当前故障类型 分合闸不彻底下每组波形的能量熵
            energyEntropy = Energy.get_entropy()

            # 4、计算svd奇异熵
            Svd = svd_entropy(emd_data.T)  # 实例化对象
            # svd = Svd.get_singular_value() # 调用方法求奇异值
            # print(2,len(svd))
            proba_svd = Svd.get_se()  # 调用方法求归一化后的值
            if len(proba_svd) >= 6:
                if len(proba_svd) == 6:
                    Proba_svd.append(proba_svd)
                else:
                    Proba_svd.append(proba_svd[0:6])
            # 得到当前故障类型 分合闸不彻底下每组波形的奇异熵
            svdEntropy = Svd.get_entropy()

            # 5、计算近似熵(计算最慢)
            # 5.1、普通函数计算
            # approximateEntropy1 = Approximate_Entropy(column_data.values)
            # 5.2、库函数计算
            # approximateEntropy2 = vectapen.apen(column_data.values,2,0.2*np.std(column_data.values))
            # output : (approximateEntropy1 ,approximateEntropy2)
            # 1.7131539393168391 0.4596967369929981 ; 2.0103529261956163 0.39715366020559134
            # 1.358543606352516 0.3694190565191563  ; 1.9608186556448284 0.35132909888094394
            # 根据样本熵的大小在0.8~0.3之间 所以近似熵这里使用库函数计算 满足样本熵>近似熵
            approximateEntropy = vectapen.apen(column_data.values, 2, 0.2 * np.std(column_data.values))

            # 6、计算样本熵
            std_ts = np.std(column_data.values)
            sampleEntropys = ent.sample_entropy(column_data.values, 5, 0.2 * std_ts)
            # output : sampleEntropys : [0.81099846 0.10495251 0.07954186 0.06707372 0.05662986]
            # 取第一个值为样本熵
            sampleEntropy = sampleEntropys[0]

            # 7、计算模糊熵(很慢,且数据)
            # fuzzyEntropy = Fuzzy_Entropy(column_data.values)
            # output : 0.030451356450480804

            # 8、计算排列熵
            PE = permutation_entropy(column_data.values)  # 实例化
            permutationEntropy = PE.get_Entropy(5, 3)
            # output :0.6214911644943121 ~ 0.46054127316779775

            # 计算时域特征
            # 1、平均值
            waveform_mean = np.mean(column_data.values)
            # 2、标准差
            waveform_std = np.std(column_data.values)
            # 3、最大值
            waveform_max = np.max(column_data.values)
            # 4、最小值
            waveform_min = np.min(column_data.values)
            # 5、峰值
            waveform_peak = max(abs(waveform_max), abs(waveform_min))
            # 6、均方根值
            waveform_rms = math.sqrt(sum([x ** 2 for x in column_data.values]) / len(column_data.values))
            # 7、峰值因子 = 峰值 / 均方根值
            waveform_crest_factor = waveform_peak / waveform_rms
            # 8、偏度
            waveform_skew = np.mean((column_data.values - waveform_mean) ** 3) \
                            / pow(waveform_std, 3)
            # 9、峭度
            waveform_kurt = np.mean((column_data.values - waveform_mean) ** 4) \
                            / pow(np.var(column_data.values), 2)

            # 得到一个pd.Series 存放每组波形的熵和时域特征
            # 依次是互相关熵、能量熵、奇异熵、近似熵、样本熵、排列熵、峰态系数、偏度、峭度
            features_series = pd.Series([corrEntropy, energyEntropy, svdEntropy,
                                         approximateEntropy, sampleEntropy, permutationEntropy
                                            , waveform_crest_factor, waveform_skew, waveform_kurt])
            # 循环拼接每组波形的特征
            features_dataframe = pd.concat([features_dataframe, features_series], axis=1, ignore_index=True)
        # 数据特征 此时为 特征数 × 波形数 希望变成波形数 × 特征数 即 11&10 × 9
        features_dataframe = features_dataframe.T
        # 添加表头
        features_dataframe.columns = ["corrEntropy", "energyEntropy", "svdEntropy", "approximateEntropy",
                                      "sampleEntropy", "permutationEntropy", "crest_factor",
                                      "skewness", "kurtosis"]
        proba_svd_mean = np.array(Proba_svd).mean(axis=0)
        proba_energy_mean = np.array(Proba_energy).mean(axis=0)
        proba_corr_mean = np.array(Proba_corr).mean(axis=0)
        self.plot_energy_svd_corr(proba_energy_mean, proba_svd_mean, proba_corr_mean, title)
        return features_dataframe, proba_energy_mean, proba_svd_mean, proba_corr_mean

    # 80、比较不同故障类型的energy
    def compare_energy(self,proba_energy_mean1, proba_energy_mean2, proba_energy_mean3, proba_energy_mean4,
                       proba_energy_mean5, proba_energy_mean6, proba_energy_mean7, proba_energy_mean8):
        imf_len = np.arange(1, len(proba_energy_mean1) + 1, 1)
        plt.plot(imf_len, proba_energy_mean1, "-o", label="分闸不彻底", linewidth=1.5)
        plt.plot(imf_len, proba_energy_mean2, "-o", label="正常分闸", linewidth=1.5)
        plt.plot(imf_len, proba_energy_mean3, "-o", label="正常合闸", linewidth=1.5)
        plt.plot(imf_len, proba_energy_mean4, "-o", label="虚假合闸", linewidth=1.5)
        plt.plot(imf_len, proba_energy_mean5, "-o", label="储能弹簧卡涩", linewidth=1.5)
        plt.plot(imf_len, proba_energy_mean6, "-o", label="操作机构卡涩", linewidth=1.5)
        plt.plot(imf_len, proba_energy_mean7, "-o", label="合闸螺栓松动", linewidth=1.5)
        plt.plot(imf_len, proba_energy_mean8, "-o", label="分闸螺栓松动", linewidth=1.5)
        plt.xlabel("imf分量")
        plt.ylabel("归一化能量值")
        plt.title("不同故障振动信号归一化能量值的对比")
        plt.legend()
        plt.show()

    # 81、比较不同故障类型的corr
    def compare_corr(self,proba_corr_mean_1, proba_corr_mean_2, proba_corr_mean_3, proba_corr_mean_4,
                     proba_corr_mean_5, proba_corr_mean_6, proba_corr_mean_7, proba_corr_mean_8):
        imf_len = np.arange(1, len(proba_corr_mean_1) + 1, 1)
        plt.plot(imf_len, proba_corr_mean_1, "-o", label="分闸不彻底", linewidth=1.5)
        plt.plot(imf_len, proba_corr_mean_2, "-o", label="正常分闸", linewidth=1.5)
        plt.plot(imf_len, proba_corr_mean_3, "-o", label="正常合闸", linewidth=1.5)
        plt.plot(imf_len, proba_corr_mean_4, "-o", label="虚假合闸", linewidth=1.5)
        plt.plot(imf_len, proba_corr_mean_5, "-o", label="储能弹簧卡涩", linewidth=1.5)
        plt.plot(imf_len, proba_corr_mean_6, "-o", label="操作机构卡涩", linewidth=1.5)
        plt.plot(imf_len, proba_corr_mean_7, "-o", label="合闸螺栓松动", linewidth=1.5)
        plt.plot(imf_len, proba_corr_mean_8, "-o", label="分闸螺栓松动", linewidth=1.5)
        plt.xlabel("imf分量")
        plt.ylabel("归一化互相关系数")
        plt.title("不同故障振动信号归一化互相关系数的对比")
        plt.legend()
        plt.show()

    # 82、比较不同故障类型的svd
    def compare_svd(self,proba_svd_mean_1, proba_svd_mean_2, proba_svd_mean_3, proba_svd_mean_4,
                    proba_svd_mean_5, proba_svd_mean_6, proba_svd_mean_7, proba_svd_mean_8):
        imf_len = np.arange(1, len(proba_svd_mean_1) + 1, 1)
        plt.plot(imf_len, proba_svd_mean_1, "-o", label="分闸不彻底", linewidth=1.5)
        plt.plot(imf_len, proba_svd_mean_2, "-o", label="正常分闸", linewidth=1.5)
        plt.plot(imf_len, proba_svd_mean_3, "-o", label="正常合闸", linewidth=1.5)
        plt.plot(imf_len, proba_svd_mean_4, "-o", label="虚假合闸", linewidth=1.5)
        plt.plot(imf_len, proba_svd_mean_5, "-o", label="储能弹簧卡涩", linewidth=1.5)
        plt.plot(imf_len, proba_svd_mean_6, "-o", label="操作机构卡涩", linewidth=1.5)
        plt.plot(imf_len, proba_svd_mean_7, "-o", label="合闸螺栓松动", linewidth=1.5)
        plt.plot(imf_len, proba_svd_mean_8, "-o", label="分闸螺栓松动", linewidth=1.5)
        plt.xlabel("imf分量")
        plt.ylabel("归一化奇异值")
        plt.title("不同故障振动信号归一化奇异值的对比")
        plt.legend()
        plt.show()

    # 83、得到降维后的pca特征(2维)、主成分散点图、方差贡献率
    def compress_features_to_pca(self,path, isSave):
        '''
        path = "C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/振动信号归一化合并特征.csv"
        compress_features_to_pca(path,False)
        :param path: 振动信号得到的83×9的特征矩阵
        :param isSave: 是否保存数据
        :return: 降维后的pca特征(2维)、主成分散点图、方差贡献率
        '''
        X = pd.read_csv(path, encoding='gbk')
        model = PCA(n_components=2)
        X_Features = X.iloc[:, 0:-1]  # 没有分类标签的纯数据
        X_new = model.fit_transform(X_Features)  # 训练并转换
        X_new_dataframe = pd.DataFrame(X_new)
        Category = X.iloc[:, -1]  # 取出标签
        X_new_category = pd.concat([X_new_dataframe, Category], axis=1, ignore_index=True)  # 按列拼接
        X_new_category.columns = ['PCA1', 'PCA2', 'Category']  # 重命名
        # 保存数据
        if isSave:
            X_new_category.to_csv(r"C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件\振动信号归一化PCA特征.csv", index=False)
        noise_variance = model.noise_variance_
        score = model.score(X_Features)
        singular_value = model.singular_values_
        # 绘制主成分降维后的散点图
        # 1、可以直接使用 上边的X_new_category 2、预测效果不好的话可以使用下边这个
        # X_new_category = pd.read_csv(r"C:\Users\chenbei\Desktop\陈北个人论文\图源数据及其文件\vibration_pca_features_data.csv",encoding='gbk')
        plt.figure(figsize=(7.8, 3.8), dpi=600)
        fig = sns.scatterplot(data=X_new_category, x='PCA1', y='PCA2', hue='Category')  # 主成分降维后的特征量的散点图可视化
        fig.set_xlabel('PCA1', fontsize=10.5)  # 设置字体大小
        fig.set_ylabel('PCA2', fontsize=10.5)
        plt.text(0, 1.0, "噪声协方差 : " + str(round(noise_variance, 5)), fontsize=10.5, verticalalignment='center',
                 horizontalalignment='center', family='SimHei')
        plt.text(0, 1.5, "似然平均值 : " + str(round(score, 5)), fontsize=10.5, verticalalignment='center',
                 horizontalalignment='center', family='SimHei')
        plt.title('主成分散点图(n_conponents=2)')
        plt.tick_params(axis='both', which='major', labelsize=10.5)  # 设置刻度
        plt.tight_layout()
        plt.show()

        '''取出4个也是一样的，为了得到主成分方差贡献率，专门一段代码用于绘制方差贡献率图'''
        model1 = PCA(n_components=4)
        X_new1 = model1.fit_transform(X_Features)
        X_new1_dataframe = pd.DataFrame(X_new1)

        np.set_printoptions(precision=5)
        ratio1 = np.around(model1.explained_variance_ratio_, 5)  # 主成分降维的方差贡献率
        component_nums = model1.n_components_
        plt.figure(figsize=(7.8, 3.8), dpi=600)
        reats = plt.bar(range(component_nums), ratio1)  # 所有柱子
        plt.ylabel('百分比')
        plt.title('主成分方差贡献率')
        plt.xlabel('维度')
        plt.xticks(ticks=[0, 1, 2, 3], labels=['PCA1', 'PCA2', 'PCA3', 'PCA4'])
        plt.tick_params(axis='both', which='major', labelsize=10.5)
        for reat in reats:  # 每一个柱子循环标注数值大小
            reat_height = reat.get_height()  # 获取柱子高度
            plt.text(reat.get_x() + reat.get_width() / 2, reat_height, str(reat_height), size=10.5, ha='center',
                     va='bottom')
        plt.tight_layout()
        plt.show()

    '''5、特征预测用到的部分函数'''
    # 84、svm的学习率函数
    def my_learning_curve(self,estimator, title, X, y, ylim=None,
                           cv=None, train_size=np.linspace(.1, 1.0, 5)):
        '''
        :param estimator: 计算指定的学习模型estimator在不同大小的训练集经过交叉验证后的训练的分和测试得分
        :param title: 标题
        :param X: 特征矩阵 : 样本数×维数
        :param y: 标签 : 行向量
        :param ylim: y轴限制区间
        :param cv: 交叉验证次数(默认3-fold), int指定"折"的数量 , 一个产生train/test划分的迭代器对象
        :param train_size: 指定训练集子集的比例,默认np.linspace(.1,1.0,5)
        :return: 不同训练集比例的学习率曲线
        '''
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel("训练集样本个数")
        plt.ylabel("得分")
        train_sizes, train_scores, test_scores, fittime, scoretime = learning_curve(
            estimator, X, y, cv=cv, n_jobs=1, train_sizes=train_size, return_times=True)
        # print(train_scores)
        # 返回已用于生成学习曲线的训练示例数、训练集得分、测试集得分、拟合时间、得分时间
        # cv规定了每次交叉验证的次数和比例
        # 返回的训练测试集得分矩阵为n_ticks × n_folds , 即列数取决于n_splits ,行数取决于train_size的个数
        # 表示训练子集的比例依次为0.1,0.109,0.118....,1时,每次交叉验证的得分
        # 例如cv中已经确定test_size为0.2,那么应当有0.8*241=192个样本作为训练集
        # 这192个训练样本中的0.1即只有19个样本真正为训练集,对(241-192=49个测试样本进行预测),进行100次交叉验证
        # 直到最后一次,192）0.9909个样本全部作为训练集对测试样本交叉验证
        # 也就是说train_size先控制子训练集的比例,test_size再控制最终的测试集
        # 按列平均将矩阵变为向量
        train_scores_mean = np.mean(train_scores, axis=1)  # 每一次交叉验证训练集得分的平均值
        train_scores_std = np.std(train_scores, axis=1)  # 每一次交叉验证训练集得分的标准差
        test_scores_mean = np.mean(test_scores, axis=1)  # 每一次交叉验证测试集得分的平均值
        test_scores_std = np.std(test_scores, axis=1)  # 每一次交叉验证测试集得分的标准差
        plt.grid()
        # 两个函数之间的区域用黄色填充
        # plt.fill_between(x, y1, y2, facecolor="yellow")
        '''为了画出带状区域,使用训练得分平均值减去或加上训练得分标准差,这两条曲线之间填充颜色'''
        # 训练集上下包络线
        plt.fill_between(
            train_sizes,
            train_scores_mean -
            train_scores_std,
            train_scores_mean +
            train_scores_std,
            alpha=0.1,
            color="r")
        # 测试集上下包络线
        plt.fill_between(
            train_sizes,
            test_scores_mean -
            test_scores_std,
            test_scores_mean +
            test_scores_std,
            alpha=0.1,
            color="g")
        # 测试集/训练集 主曲线
        plt.plot(
            train_sizes,
            train_scores_mean,
            "o-",
            color="r",
            label="交叉验证-训练集得分")  # 图例可以使用label控制
        plt.plot(
            train_sizes,
            test_scores_mean,
            "o-",
            color="g",
            label="交叉验证-测试集得分")
        plt.legend(loc="best")
        plt.axis('tight')
        plt.show()
        return plt


    # 85、绘制学习率
    def plot_my_learning_curve(self,X, y, title, n_splits=10, train_size=0.9):
        cv = ShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=0)
        train_sizes = np.linspace(0.1, 1.0, 10)  # 控制训练集的子集比例
        # title = "线性核Linear不同训练集比例的10折交叉验证得分(固定正则化系数C=0.1)"
        estimator = svm.SVC(kernel='linear', probability=True)
        self.my_learning_curve(estimator, title, X, y, ylim=(0.7, 1.0), cv=cv, train_size=train_sizes)

    # 86、svm的评价指标函数
    def print_evaluation_index(self,X_test, y_test, y_pre, clf,path, confusion_matrix=False, title=None,isSave=False):
        '''返回评价指标'''
        # 1、准确度 ACU
        # ACU = clf.score(X_test,y_test) # 返回给定测试数据和标签上的平均准确度
        ACU = metrics.accuracy_score(y_test, y_pre)  # 直接引用方法accuracy_score是相同的
        print('平均准确率ACU为:', ACU)
        # print('准确率得分为:',ACU_Score)
        # 2、精准率 PRECISON
        Precison_Score = metrics.precision_score(
            y_test, y_pre, average='weighted')  # 精准率
        print('精准率Precison为:', Precison_Score)
        # 3、召回率 RECALL
        Recall = metrics.recall_score(y_test, y_pre, average='weighted')
        print('召回率Recall为:', Recall)
        # 4、F1 得分
        F1 = metrics.f1_score(y_test, y_pre, average='weighted')
        print('F1得分为:', F1)
        # 以前的参数输入值是真实标签和预测标签,除了平均准确率外对于多类问题必须指定avarage的值
        # 5、混淆矩阵
        CM = metrics.confusion_matrix(y_test, y_pre)
        # 6、绘制混淆矩阵
        if confusion_matrix:
            plt.figure(figsize=(4.4, 3.8))
            metrics.plot_confusion_matrix(clf, X_test, y_test)
            plt.xlabel('预测标签', fontproperties=font_set)
            # plt.ylabel('实际标签',fontsize=10.5,fontname='宋体') # 会出现乱码
            plt.ylabel('实际标签', fontproperties=font_set)
            plt.title(f'{title}的混淆矩阵', fontproperties=font_set)
            plt.show()
        # 7、最大误差
        # max_error = metrics.max_error(y_test,clf.predict(X_test))
        # print('预测最大误差为:',max_error) = 7
        # 8、方均根误差
        RMS = metrics.mean_squared_error(y_test, y_pre)
        print('方均根误差RMS为:', RMS)
        # 9、显示主要的指标报告
        # report = metrics.classification_report (y_test,y_pre ,output_dict=True) # 返回字典
        # 10、汉明损失是被错误预测的标签分数
        hammingloss = metrics.hamming_loss(y_test, y_pre)
        print('Hamming损失为:', hammingloss)
        # 11、多重混淆矩阵
        # MULCM = metrics.multilabel_confusion_matrix(y_test,clf.predict(X_test))
        # Matthews相关系数
        matthews = metrics.matthews_corrcoef(y_test, y_pre)
        print('Matthews相关系数为:', matthews)
        # 12、R2系数
        R2 = metrics.r2_score(y_test, y_pre)
        print('R2系数为:', R2)
        # 13、平衡精度BAS
        BAS = metrics.balanced_accuracy_score(y_test, y_pre)
        print('平衡精度BAS为:', BAS)
        # 14、精度和召回率的加权谐波平均值F_beta
        F_beta = metrics.fbeta_score(
            y_test,
            y_pre,
            beta=0.5,
            average='weighted')  # beta 确定组合分数中的召回权重
        print('加权谐波平均值F_beta为:', F_beta)
        # 15、Jaccard相似系数
        Jaccard = metrics.jaccard_score(y_test, y_pre, average='weighted')
        print('Jaccard相似系数为:', Jaccard)
        # 16、调整互信息 AMI 输入参数是labels_true, labels_pred
        AMI = metrics.adjusted_mutual_info_score(y_test, y_pre)
        print('调整互信息AMI为:', AMI)
        # 17、标准化互信息NMI
        NMI = metrics.normalized_mutual_info_score(y_test, y_pre)
        print('标准化互信息NMI为:', NMI)
        # 18、方差回归得分EVS
        EVS = metrics.explained_variance_score(y_test, y_pre)
        print('方差回归得分EVS为:', EVS)
        # 19、标签排名平均精度LRAP
        # LRAP = metrics.label_ranking_average_precision_score(y_test,clf.predict(X_test)) 不支持多重
        # 20、Fowlkes-Mallows指数（FMI）定义为精度和召回率之间的几何平均值
        FMI = metrics.fowlkes_mallows_score(y_test, y_pre)
        print('Fowlkes-Mallows指数FMI为:', FMI)
        Evaluation_index = [ACU, Precison_Score, Recall, F1,
                            RMS, hammingloss, matthews, R2, BAS, F_beta, Jaccard,
                            AMI, NMI, EVS, FMI]
        Evaluation_index = pd.DataFrame(Evaluation_index)
        Evaluation_index = Evaluation_index.T
        Evaluation_index.columns = [
            '平均准确率ACU',
            '精准率Precision',
            '召回率Recall',
            'F1得分',
            '方均根误差RMS',
            '汉明损失HLoss',
            'Matthews相关系数',
            'R2系数',
            '平衡精度BAS',
            '加权谐波平均值F_beta',
            '相似系数Jaccard',
            '调整互信息AMI',
            '标准化互信息NMI',
            '方差回归得分EVS',
            'Fowlkes-Mallows指数FMI']
        if isSave :
            Evaluation_index.to_csv(path,index=False)
        return Evaluation_index

    # 87、使用留出法绘制核密度估计图
    def plot_kernel_density_estimation(self,X, y, maxiter, train_size):
        EIK = pd.DataFrame()
        starttime = time()
        for i in range(maxiter):
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, train_size=train_size)  # 训练数据比例7成
            clf = svm.SVC(kernel='linear', C=1, probability=True)
            clf.fit(X_train, y_train)  # 训练模型
            y_pre = clf.predict(X_test)
            Evaluation_index = self.print_evaluation_index(
                X_test, y_test, y_pre, clf, confusion_matrix=False)
            EIK = pd.concat([EIK, Evaluation_index], axis=0, ignore_index=True)
        endtime = time()
        Time = (endtime - starttime) / 100
        # bw_method 太小过拟合,太大欠拟合 ,ind指定x轴位置转折 ind=[0.81,0.85,0.91,0.95,0.97,0.98]
        ACU = EIK['平均准确率ACU']
        fig, ax = plt.subplots()
        ax.grid(axis='y')
        ACU.plot.kde(ax=ax, linewidth=2.5, bw_method=0.3)
        ACU.plot.hist(density=True, bins=12, color='c', ax=ax)
        ax.set_xlabel('准确率')
        ax.set_ylabel('频数')
        ax.axis('tight')
        labels = ["核密度估计图", "直方图"]
        ax.legend(labels, loc='upper left')
        ax1 = ax.twinx()
        ax1.plot(ACU, ACU, 'b-v', linewidth=0.5)
        meanACU = np.mean(ACU)
        ax1.plot([min(ACU), 1.0], [meanACU, meanACU], 'r--')
        ax1.text((min(ACU) + 1.0) / 2, meanACU + 0.005, "平均值 : " + str(round(meanACU, 5)),
                 fontsize=14, verticalalignment='center', horizontalalignment='center')
        ax1.text((min(ACU) + 1.0) / 2, min(ACU) + 0.006, "训练时间 : " + str(round(Time, 5)) +
                 ' s', verticalalignment='center', horizontalalignment='center', fontsize=14)
        ax1.axis('tight')
        ax1.legend(('折线图', '平均值'), loc='center left')
        ax1.fill_betweenx(
            np.arange(
                min(ACU) -
                0.05,
                1.02,
                0.05),
            1.0,
            1.02,
            alpha=0.2,
            color="r",
            lw=2)
        ax1.set_xlim(min(ACU), 1.02)
        ax1.set_ylim(min(ACU), 1.0)
        ax1.text(
            (1.0 + 1.02) / 2,
            (min(ACU) + 1) / 2,
            "区域不存在",
            fontsize=10.5,
            family='SimHei',
            verticalalignment='center',
            horizontalalignment='center')
        plt.title('线性核Linear留出法预测准确率分布图(训练次数:100次)')
        fig.tight_layout()
        plt.show()

    # 88、比较不同正则化系数alpha交叉验证得分 固定交叉验证次数
    def compare_nfold_cross_validation_score(self,X, y, n_folds):
        C_s = np.logspace(-30, 30, 100)  # 10^(-10) ~ 10^0 , 分成十个,默认基准值10为底数
        # C_s = np.linspace(0.000001,1,100,endpoint=False)
        # n_folds = 8
        scores = list()  # 存放不同正则化系数的得分
        scores_std = list()
        clf = svm.SVC(kernel='linear', probability=True)
        for C in C_s:
            clf.C = C
            this_scores = cross_val_score(
                clf, X, y, cv=n_folds, n_jobs=1)  # n_jobs用于进行计算的CPU数量
            scores.append(np.mean(this_scores))
            scores_std.append(np.std(this_scores))
        # X轴采用对数刻度 , 解决图例一张图的方法
        ax1, = plt.semilogx(C_s, scores)  # 绘制得分
        ax2, = plt.semilogx(C_s, np.array(scores) +
                            np.array(scores_std), 'r--')  # 绘制得分的上包络线
        ax3, = plt.semilogx(C_s, np.array(scores) -
                            np.array(scores_std), 'b--')  # 绘制得分的下包络线
        plt.fill_between(
            C_s,
            np.array(scores) +
            np.array(scores_std),
            np.array(scores) -
            np.array(scores_std),
            alpha=0.1,
            color="r")  # 着色
        plt.legend([ax1, ax2, ax3], ['交叉验证得分', '上包络线', '下包络线'], loc='best')
        # 解决方案
        ax = plt.gca()
        locs, labels = plt.yticks()  # 刻度的位置和它使用的数值
        # 每个位置上使用新的刻度值,可以将小数刻度变为指数刻度
        plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
        for tick in ax.xaxis.get_major_ticks():  # 获取图片坐标轴的主刻度,循环设置格式 解决指数坐标不能显示负号的问题
            tick.label1.set_fontproperties('stixgeneral')
        # 以上代码复制粘贴即可
        plt.ylabel('交叉验证得分')
        plt.xlabel("正则化系数C")
        plt.title(f"线性核Linear不同正则化系数的{n_folds}折交叉验证得分趋势图")
        plt.axis('tight')
        plt.show()

    # 89、比较不同gamma系数的交叉验证得分
    def compare_gamma_cross_validation_score(self,X, y):
        param_range = np.logspace(-30, 30, 100)
        train_scores, test_scores = validation_curve(svm.SVC(
            kernel='linear'), X, y, param_name="gamma", param_range=param_range, cv=10, scoring="accuracy", n_jobs=1)
        train_scores_mean = np.mean(train_scores, axis=1)  # 训练集得分
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)  # 测试集得分
        test_scores_std = np.std(test_scores, axis=1)
        # gamma是选择RBF函数作为kernel后，该函数自带的一个参数 gamma越大，支持向量越少，gamma值越小，支持向量越多。支持向量的个数影响训练与预测的速度
        # gamma 与 高斯部分方差成反比,gamma越大方差越小,分布越集中,训练样本准确率很高但是泛化能力差 gamma=1/(2*方差)
        plt.title(r"线性核Linear不同$\gamma$系数的交叉验证得分趋势图")
        plt.xlabel(r"内核系数$\gamma$")
        plt.ylabel("得分")
        plt.ylim(0.0, 1.0)
        lw = 1.5
        # 训练集 主曲线
        ax1, = plt.semilogx(param_range, train_scores_mean,
                            label="交叉验证-训练集得分", color="r", lw=lw)
        ax2, = plt.semilogx(param_range, np.array(train_scores_mean) +
                            np.array(train_scores_std), 'b--', label='训练集上包络线', lw=0.5)  # 训练集上包络线
        ax3, = plt.semilogx(param_range, np.array(train_scores_mean) -
                            np.array(train_scores_std), 'b--', label='训练集下包络线', lw=0.5)  # 训练集下包络线
        # 训练集 上下包络线
        plt.fill_between(
            param_range,
            train_scores_mean -
            train_scores_std,
            train_scores_mean +
            train_scores_std,
            alpha=0.2,
            color="r",
            lw=lw)
        # 测试集 主曲线
        ax4, = plt.semilogx(param_range, test_scores_mean,
                            label="交叉验证-测试集得分", color="g", lw=lw)
        ax5, = plt.semilogx(param_range, np.array(test_scores_mean) +
                            np.array(test_scores_std), 'y--', label='测试集上包络线', lw=0.5)  # 测试集上包络线
        ax6, = plt.semilogx(param_range, np.array(test_scores_mean) -
                            np.array(test_scores_std), 'y--', label='测试集下包络线', lw=0.5)  # 测试集下包络线
        # 测试集 上下包络线
        plt.fill_between(
            param_range,
            test_scores_mean -
            test_scores_std,
            test_scores_mean +
            test_scores_std,
            alpha=0.2,
            color="g",
            lw=lw)
        minscore = min(
            np.min(
                np.array(train_scores_mean) -
                np.array(train_scores_std)),
            np.min(
                np.array(test_scores_mean) -
                np.array(test_scores_std)))
        plt.ylim((minscore, 1))
        plt.legend(loc='best')
        # 不显示指数负号解决方案
        ax = plt.gca()
        locs, labels = plt.yticks()  # 刻度的位置和它使用的数值
        # 每个位置上使用新的刻度值,可以将小数刻度变为指数刻度
        plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
        for tick in ax.xaxis.get_major_ticks():  # 获取图片坐标轴的主刻度,循环设置格式 解决指数坐标不能显示负号的问题
            tick.label1.set_fontproperties('stixgeneral')
        plt.show()

    # 90、绘制奈斯特罗和蒙特卡洛估计
    def plot_rbf_sampler_nystroem(self,y, path, gamma=0.2, train_size=0.8):
        '''
        path = r"C:/Users\chenbei\Desktop\陈北个人论文\图源数据及其文件/Circuit_Breaker_FirstLevelLabel.
        plot_rbf_sampler_nystroem(y,path=path)
        :param y: 数据标签
        :param path: pca数据路径
        :param gamma: gamma
        :param train_size: 训练及比例
        :return: 奈斯特罗和随机傅里叶估计
        '''

        X = pd.read_csv(path, encoding='gbk')
        X = X.drop(['Category'], axis=1)  # 删除标签列
        X = X.values  # 划分训练和测试的参数是数组格式
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
        n_spilit = len(X_train)
        n_spilit_size = np.linspace(2, n_spilit, n_spilit - 1)  # 2,3,4,...192 至少2份
        sample_sizes = np.arange(1, n_spilit)
        kernel_svm = svm.SVC(gamma=gamma)  # 不指定默认是rbf
        linear_svm = svm.LinearSVC()  # 与指定'linear'是一样的
        Linear_time = []
        Kernel_time = []
        Linear_score = []
        Kernel_score = []
        starttime = time()
        for i in range(len(n_spilit_size)):  # 依次使用训练集的2,3,4,...192个样本
            # 以下四行程序是为了保证每次使用训练集其中的i个样本也是随机的
            state = np.random.get_state()  # 必须先打乱原来的数据集合标签集,否则初始的有序状态会导致交叉验证预测率很低
            np.random.shuffle(X_train)
            np.random.set_state(state)  # 保证样本和标签以相同的规律被打乱
            np.random.shuffle(y_train)
            # print(int(n_spilit_size[i]))
            X_train_son = X_train[0:int(n_spilit_size[i]), :]  # 前 i 行 作为训练集 的子集
            y_train_son = y_train[0:int(n_spilit_size[i])]  # 前 i 行 作为训练集 的子集
            Kernel_svm_time = time()  # rbf核函数的程序开始时间
            kernel_svm.fit(X_train_son, y_train_son)  # 训练集拟合
            kernel_svm_score = kernel_svm.score(X_test, y_test)  # 测试集得分
            kernel_svm_time = time() - Kernel_svm_time  # rbf核函数运行消耗的时间
            Linear_svm_time = time()  # 线性核函数的程序开始时间
            linear_svm.fit(X_train, y_train)  # 训练集拟合
            linear_svm_score = linear_svm.score(X_test, y_test)  # 测试集得分
            linear_svm_time = time() - Linear_svm_time  # 线性核函数运行消耗的时间
            Linear_score.append(linear_svm_score)  # 存放不同训练子集数量预测测试集的时间和分数
            Linear_time.append(linear_svm_time)
            Kernel_score.append(kernel_svm_score)
            Kernel_time.append(kernel_svm_time)
        '''线性核 : 从内核近似创建管道'''
        feature_map_fourier = RBFSampler(gamma=gamma, random_state=1)  # 傅里叶变换的蒙特卡洛近似
        feature_map_nystroem = Nystroem(
            gamma=gamma, random_state=1)  # 使用数据的子集为任意内核构造一个近似特征图
        fourier_approx_svm_rbf = pipeline.Pipeline(
            [("feature_map", feature_map_fourier), ("svm", svm.SVC(gamma=gamma))])  # 基于RBF构建第一个管道
        nystroem_approx_svm_rbf = pipeline.Pipeline(
            [("feature_map", feature_map_nystroem), ("svm", svm.SVC(gamma=gamma))])  # 基于RBF构建第二个管道
        fourier_approx_svm_linear = pipeline.Pipeline(
            [("feature_map", feature_map_fourier), ("svm", svm.LinearSVC())])  # 基于Linear构建第一个管道
        nystroem_approx_svm_linear = pipeline.Pipeline(
            [("feature_map", feature_map_nystroem), ("svm", svm.LinearSVC())])  # 基于Linear构建第二个管道
        # sample_sizes = 30 * np.arange(1, 10)
        fourier_scores_rbf = []  # 分别存放fourier和nystroem内核估计的分数和消耗时间
        nystroem_scores_rbf = []
        fourier_times_rbf = []
        nystroem_times_rbf = []
        fourier_scores_linear = []  # 分别存放fourier和nystroem内核估计的分数和消耗时间
        nystroem_scores_linear = []
        fourier_times_linear = []
        nystroem_times_linear = []
        for D in sample_sizes:
            # 基于RBF
            fourier_approx_svm_rbf.set_params(
                feature_map__n_components=D)  # 管道1设置参数 ,D不能超过训练集的数量
            nystroem_approx_svm_rbf.set_params(feature_map__n_components=D)
            start = time()
            nystroem_approx_svm_rbf.fit(X_train, y_train)  # rbf的nystroem时间
            nystroem_times_rbf.append(time() - start)
            start = time()
            fourier_approx_svm_rbf.fit(X_train, y_train)  # rbf的fourier时间
            fourier_times_rbf.append(time() - start)
            fourier_score = fourier_approx_svm_rbf.score(
                X_test, y_test)  # rbf的nystroem得分
            nystroem_score = nystroem_approx_svm_rbf.score(
                X_test, y_test)  # rbf的fourier得分
            nystroem_scores_rbf.append(nystroem_score)
            fourier_scores_rbf.append(fourier_score)
            # 基于Linear
            fourier_approx_svm_linear.set_params(
                feature_map__n_components=D)  # 管道1设置参数 ,D不能超过训练集的数量
            nystroem_approx_svm_linear.set_params(feature_map__n_components=D)
            start = time()
            nystroem_approx_svm_linear.fit(X_train, y_train)  # rbf的nystroem时间
            nystroem_times_linear.append(time() - start)
            start = time()
            fourier_approx_svm_linear.fit(X_train, y_train)  # rbf的fourier时间
            fourier_times_linear.append(time() - start)
            fourier_score = fourier_approx_svm_linear.score(
                X_test, y_test)  # rbf的nystroem得分
            nystroem_score = nystroem_approx_svm_linear.score(
                X_test, y_test)  # rbf的fourier得分
            nystroem_scores_linear.append(nystroem_score)
            fourier_scores_linear.append(fourier_score)

        plt.figure(figsize=(12, 8.4))
        accuracy = plt.subplot(211)  # 准确率
        timescale = plt.subplot(212)  # 耗时
        # 基于不同核函数的两种估计的时间和准确率
        accuracy.plot(sample_sizes, nystroem_scores_rbf, label="基于RBF核的奈斯特罗姆近似-准确率")
        timescale.plot(
            sample_sizes,
            nystroem_times_rbf,
            '--',
            label='基于RBF核的奈斯特罗姆近似-模型训练时间')
        accuracy.plot(sample_sizes, fourier_scores_rbf, label="基于RBF核的蒙特卡洛近似-准确率")
        timescale.plot(
            sample_sizes,
            fourier_times_rbf,
            '--',
            label='基于RBF核的蒙特卡洛近似-模型训练时间')

        accuracy.plot(
            sample_sizes,
            nystroem_scores_linear,
            label="基于Linear核的奈斯特罗姆近似-准确率")
        timescale.plot(sample_sizes, nystroem_times_linear,
                       '--', label='基于Linear核的奈斯特罗姆近似-模型训练时间')
        accuracy.plot(
            sample_sizes,
            fourier_scores_linear,
            label="基于Linear核的蒙特卡洛近似-准确率")
        timescale.plot(
            sample_sizes,
            fourier_times_linear,
            '--',
            label='基于Linear核的蒙特卡洛近似-模型训练时间')
        # 没有近似的线性核和RBF核的时间和准确率
        accuracy.plot(sample_sizes, Linear_score, label="普通线性核-准确率")
        timescale.plot(sample_sizes, Linear_time, '--', label='普通线性核-模型训练时间')
        accuracy.plot(sample_sizes, Kernel_score, label="普通RBF核-准确率")
        timescale.plot(sample_sizes, Kernel_time, '--', label='普通RBF核-模型训练时间')
        # 精确的rbf和线性内核的水平线
        # accuracy.plot([sample_sizes[0], sample_sizes[-1]],[linear_svm_score, linear_svm_score], label="线性核平均准确率")
        # timescale.plot([sample_sizes[0], sample_sizes[-1]],[linear_svm_time, linear_svm_time], '--', label='线性核平均模型训练时间')
        # accuracy.plot([sample_sizes[0], sample_sizes[-1]],[kernel_svm_score, kernel_svm_score], label="RBF核平均准确率")
        # timescale.plot([sample_sizes[0], sample_sizes[-1]],[kernel_svm_time, kernel_svm_time], '--', label='RBF核平均模型训练时间')
        # 数据集维度的垂直线 = 23 这个可以观察nystroem_scores得分在第23个明显上升
        # accuracy.plot([23, 23], [min(nystroem_scores[0],fourier_scores[0]), 1], label="分界线") # 这里观察最低得分0.4794 ,即垂直线的两个点,横坐标都是4,纵坐标分别是0.47和1
        # 设置图片参数
        minscore = min(
            np.min(fourier_scores_rbf),
            np.min(nystroem_scores_rbf),
            np.min(Kernel_score),
            np.min(Linear_score),
            np.min(fourier_scores_linear),
            np.min(nystroem_scores_linear))  # 找到6种评价分数各自最小的最小那个作为y轴最小值
        maxscore = max(np.max(fourier_scores_rbf), np.max(nystroem_scores_rbf),
                       np.max(Kernel_score), np.max(Linear_score),
                       np.max(fourier_scores_linear), np.max(nystroem_scores_linear))
        accuracy.set_ylim(minscore, maxscore)
        accuracy.set_xlim(0, len(sample_sizes))
        accuracy.set_ylabel("准确率")
        accuracy.set_xlabel("训练集样本数量/个")
        accuracy.legend(loc='best')
        accuracy.set_title("基于Linear和RBF核的奈斯特罗姆近似与蒙特卡洛近似的准确率变化图")
        mintime = min(np.min(fourier_times_rbf), np.min(nystroem_times_rbf),
                      np.min(Kernel_time), np.min(Linear_time),
                      np.min(fourier_times_linear), np.min(nystroem_times_linear))
        maxtime = max(np.max(fourier_times_rbf), np.max(nystroem_times_rbf),
                      np.max(Kernel_time), np.max(Linear_time),
                      np.max(fourier_times_linear), np.max(nystroem_times_linear))
        timescale.set_ylim(mintime, maxtime)
        timescale.set_xlim(0, len(sample_sizes))
        timescale.set_xlabel("训练集样本数量/个")
        timescale.set_ylabel("模型训练时间/s")
        timescale.legend(loc='best')
        timescale.set_title("基于Linear和RBF核的奈斯特罗姆近似与蒙特卡洛近似的模型训练时间变化图")
        plt.show()
        endtime = time()
        print('程序花费时间为 : ', endtime - starttime)

    '''6、mlp得到最优模型用到的函数'''
    # 91、得到某个训练集比例的mlp基础模型预测概率 在get_mlp_acu和get_mlp_acu_trainsize中调用
    def return_mlp_neural_network(self,clf, X, y, trainsize=0.7):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=trainsize)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        numX_train = len(y_train)  # 返回训练样本数量
        sum = 0
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                sum = sum + 1
        sum = sum / len(y_pred)
        return sum, numX_train

    # 92、基础函数 : 得到maxiter次下固定训练比例的mlp的预测准确率
    def get_mlp_acu(self,clf, maxiter, X, y):
        ACU = []
        num_X_train = []
        for i in range(maxiter):  # 训练代数改变
            acu, num_x_train = self.return_mlp_neural_network(clf, X, y, trainsize=0.7)  # 这是训练一次得到的准确率
            ACU.append(acu)
            num_X_train.append(num_x_train)
        return ACU, num_X_train  # 返回样本数量和这maxter次的acu

    # 93、基础函数 : 返回不同训练比例的maxiter次的所有预测准确率
    # ChangeTrainingSampleRatio 函数中使用 用于循环给定不同训练比例
    def get_acu_trainsize(self,clf, maxiter, trainsize, X, y):
        '''
        与get_acu函数区别在于transize不是固定的0.7
        :param clf: 模型
        :param maxiter: 训练次数
        :param test_size: 固定的比例
        :return: 每次训练的准确率
        '''
        ACU = []
        num_X_train = []
        for i in range(maxiter):  # 训练代数改变
            acu, num_x_train = self.return_mlp_neural_network(clf, X, y, trainsize=trainsize)  # 这是训练一次得到的准确率
            ACU.append(acu)
            num_X_train.append(num_x_train)
        return ACU, num_X_train  # 返回样本数量和这maxter次的acu

    # 94、改变梯度优化器lbfgs、adam、sgd
    def ChangeSovler(self,sovler, X, y, title1, title2):
        if sovler == 'sgd':  # 做判断是因为只有sgd方法需要设定学习率和学习方法
            clf = MLPClassifier(activation='identity', solver=sovler, alpha=1e-1, hidden_layer_sizes=(5, 2),
                                random_state=1,
                                learning_rate='constant', learning_rate_init=0.001)  # 在这里不考虑不同学习率方法,只考虑不同权值优化方法
        else:
            clf = MLPClassifier(activation='identity', solver=sovler, alpha=1e-1, hidden_layer_sizes=(5, 2),
                                random_state=1)
        maxiter = 100
        ACU, _ = self.get_mlp_acu(clf, maxiter, X=X, y=y)
        self.PlotACU(ACU, maxiter=maxiter, title1=title1, title2=title2)

    # 95、改变激活函数 identity、logistic、tanh、relu
    def ChangeActivation(self,activation, X, y, title1, title2):
        clf = MLPClassifier(activation=activation, solver='lbfgs', alpha=1e-1, hidden_layer_sizes=(5, 2),
                            random_state=1)
        maxiter = 100
        ACU, _ = self.get_mlp_acu(clf, maxiter, X=X, y=y)
        self.PlotACU(ACU, maxiter=maxiter, title1=title1, title2=title2)

    # 96、改变正则化系数alpha
    # Alphas = [1e-5,1e-4,1e-3,1e-2,1e-1,0.5,1,5,10,20,50,80,100]
    def ChangeAlpha(self,alpha, X, y, title1, title2):
        clf = MLPClassifier(activation='identity', solver='lbfgs', alpha=alpha, hidden_layer_sizes=(5, 2),
                            random_state=1)
        maxiter = 100
        ACU, _ = self.get_mlp_acu(clf, maxiter, X=X, y=y)
        self.PlotACU(ACU, maxiter=maxiter, title1=title1, title2=title2)
        return ACU

    # 97、改变学习方法 constant or invscaling
    def ChangeLearingRate(self,learning_rate, X, y, title1, title2):
        if learning_rate == 'constant':
            clf = MLPClassifier(activation='identity', solver='sgd', learning_rate=learning_rate,
                                learning_rate_init=0.001,
                                alpha=0.1, hidden_layer_sizes=(5, 2), random_state=1)
        else:
            clf = MLPClassifier(activation='identity', solver='sgd', learning_rate=learning_rate,
                                learning_rate_init=0.001,
                                alpha=0.1, power_t=0.5, hidden_layer_sizes=(5, 2), random_state=1)
        maxiter = 100
        ACU, _ = self.get_mlp_acu(clf, maxiter, X=X, y=y)
        self.PlotACU(ACU, maxiter=maxiter, title1=title1, title2=title2)

    # 98、固定学习率方式下constant 改变学习率
    # Learning_rate_inits = [1e-5,1e-4,1e-3,1e-2,1e-1,0.5,1,5,10,20,50,80,100]
    def ChangeConstantLearningRateInit(self,learning_rate_init, X, y, title1, title2):
        clf = MLPClassifier(activation='identity', solver='sgd', learning_rate='constant',
                            learning_rate_init=learning_rate_init,
                            alpha=0.1, hidden_layer_sizes=(5, 2), random_state=1)
        maxiter = 100
        ACU, _ = self.get_mlp_acu(clf, maxiter, X=X, y=y)
        self.PlotACU(ACU, maxiter=maxiter, title1=title1, title2=title2)
        return ACU

    # 99、逆学习率时invscaling 改变学习率
    def ChangeInvscalingLearningRateInit(self,learning_rate_init, X, y, title1, title2):
        clf = MLPClassifier(activation='identity', solver='sgd', learning_rate='invscaling',
                            learning_rate_init=learning_rate_init,
                            alpha=0.1, hidden_layer_sizes=(5, 2), random_state=1)
        maxiter = 100
        ACU, _ = self.get_mlp_acu(clf, maxiter, X=X, y=y)
        self.PlotACU(ACU, maxiter=maxiter, title1=title1, title2=title2)
        return ACU

    # 100、逆学习率时invscaling 改变逆缩放指数power_t
    # power_ts = [1e-5,1e-4,1e-3,1e-2,1e-1,0.5,1,5,10,20,50,80]
    def ChangeInvscalingPowert(self,power_t, X, y, title1, title2):
        clf = MLPClassifier(activation='identity', solver='sgd', learning_rate='invscaling', learning_rate_init=0.1,
                            alpha=0.1, hidden_layer_sizes=(5, 2), random_state=1, power_t=power_t)
        maxiter = 100
        ACU, _ = self.get_mlp_acu(clf, maxiter, X=X, y=y)
        self.PlotACU(ACU, maxiter=maxiter, title1=title1, title2=title2)
        return ACU

    # 101、改变训练集比例
    def ChangeTrainingSampleRatio(self,X, y):
        trainsizes = np.arange(0.1, 1.0, 0.1)  # 最优参数下划分训练集比例
        clf = MLPClassifier(activation='identity', solver='lbfgs', alpha=0.1, hidden_layer_sizes=(5, 2), random_state=1)
        maxiter = 100
        ACUmean = []
        ACU_X = []
        Time = []
        for trainsize in trainsizes:  # 不同训练样本数
            starttime = time()
            # 对于不同训练集比例的需要返回样本数,用于绘图
            acu, acu_x = self.get_mlp_acu_trainsize(clf, maxiter, trainsize, X=X, y=y)  # 某一个训练比例,迭代maxiter次 得到相应的所有acu
            acu_mean = np.mean(acu)  # 找到每个训练比例下迭代maiter次的准确率平均值
            ACUmean.append(acu_mean)
            acu_x_mean = np.mean(acu_x)  # 100个 0.9比例 取平均或者取第1个都一样
            ACU_X.append(acu_x_mean)  # 每次训练的数量
            endtime = time()
            consumetime = endtime - starttime
            Time.append(consumetime)
        Time = np.array(Time) / maxiter  # 归算到每一次花费的时间
        self.PlotTrainingSampleRatio(ACU_X, ACUmean, Time, maxiter=maxiter)

    # 102、比较准确率
    def PlotACU(self,ACU, maxiter, title1, title2):
        plt.plot(ACU, 'c-p', linewidth=1, markersize=2, label='准确率')
        plt.plot([1, maxiter], [min(ACU), min(ACU)], 'r-o', label='准确率最小值', linewidth=1)
        plt.plot([1, maxiter], [np.mean(ACU), np.mean(ACU)], 'b-o', label='准确率平均值', linewidth=1)
        plt.fill_between(np.arange(1, maxiter, 1), np.mean(ACU) + np.std(ACU), np.mean(ACU) - np.std(ACU), alpha=0.1,
                         color='r')
        plt.text((1 + maxiter) / 2, np.mean(ACU) + 0.005, "准确率平均值 : " + str(round(np.mean(ACU), 5)),
                 horizontalalignment='center', color='b', fontsize=16)
        plt.text((1 + maxiter) / 2, min(ACU) + 0.005, "准确率最小值 : " + str(round(min(ACU), 5)), color='r',
                 horizontalalignment='center', fontsize=16)
        plt.title(f'不同{title1}MLP神经网络准确率变化图({title2})')
        plt.ylabel('准确率')
        plt.xlabel('训练次数')
        plt.legend(loc='lower left')
        plt.show()

    # 103、比较不同正则化系数
    def PlotAlpha(self,MinACU, MeanACU, Alphas):
        '''
        :param MinACU: 不同正则化系数的最小值
        :param MeanACU: 不同正则化系数的平均值
        :param Alphas: 正则化系数
        :return: 关于正则化的变化趋势
        '''
        plt.semilogx(Alphas, MinACU, 'r-p', label='准确率最小值', linewidth=1.5)
        plt.semilogx(Alphas, MeanACU, 'b-o', label='准确率平均值', linewidth=1.5)
        plt.legend(loc='lower left')
        ax = plt.gca()
        locs, labels = plt.yticks()  # 刻度的位置和它使用的数值
        plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))  # 每个位置上使用新的刻度值,可以将小数刻度变为指数刻度
        for tick in ax.xaxis.get_major_ticks():  # 获取图片坐标轴的主刻度,循环设置格式 解决指数坐标不能显示负号的问题
            tick.label1.set_fontproperties('stixgeneral')
        plt.xlabel('正则化系数alpha')
        plt.ylabel('准确率')
        plt.title('不同正则化系数MLP神经网络平均准确率变化图')
        plt.ylim(0, max(max(MinACU), max(MeanACU)))
        plt.show()

    # 104、比较固定学习率方式的学习率
    def PlotConstantLearningRateInit(self,MinACU, MeanACU, Learning_rate_inits):
        plt.semilogx(Learning_rate_inits, MinACU, 'r-p', label='准确率最小值', linewidth=1.5)
        plt.semilogx(Learning_rate_inits, MeanACU, 'b-o', label='准确率平均值', linewidth=1.5)
        plt.legend(loc='lower left')
        ax = plt.gca()
        locs, labels = plt.yticks()  # 刻度的位置和它使用的数值
        plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))  # 每个位置上使用新的刻度值,可以将小数刻度变为指数刻度
        for tick in ax.xaxis.get_major_ticks():  # 获取图片坐标轴的主刻度,循环设置格式 解决指数坐标不能显示负号的问题
            tick.label1.set_fontproperties('stixgeneral')
        plt.xlabel('学习率')
        plt.ylabel('准确率')
        plt.title('不同学习率MLP神经网络平均准确率变化图(SGD-Constant)')
        plt.ylim(0, max(max(MinACU), max(MeanACU)))
        plt.show()

    # 105、比较逆学习率的学习率
    def PlotInvscalingLearningRateInit(self,MinACU, MeanACU, Learning_rate_inits):
        plt.semilogx(Learning_rate_inits, MinACU, 'r-p', label='准确率最小值', linewidth=1.5)
        plt.semilogx(Learning_rate_inits, MeanACU, 'b-o', label='准确率平均值', linewidth=1.5)
        plt.legend(loc='lower left')
        ax = plt.gca()
        locs, labels = plt.yticks()  # 刻度的位置和它使用的数值
        plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))  # 每个位置上使用新的刻度值,可以将小数刻度变为指数刻度
        for tick in ax.xaxis.get_major_ticks():  # 获取图片坐标轴的主刻度,循环设置格式 解决指数坐标不能显示负号的问题
            tick.label1.set_fontproperties('stixgeneral')
        plt.xlabel('学习率')
        plt.ylabel('准确率')
        plt.title('不同学习率MLP神经网络平均准确率变化图(SGD-Invscaling)')
        plt.ylim(0, max(max(MinACU), max(MeanACU)))
        plt.show()

    # 106、比较逆学习率的逆缩放指数
    def PlotInvscalingPowert(self,MinACU, MeanACU, Power_t):
        plt.semilogx(Power_t, MinACU, 'r-p', label='准确率最小值', linewidth=1.5)
        plt.semilogx(Power_t, MeanACU, 'b-o', label='准确率平均值', linewidth=1.5)
        plt.legend(loc='lower left')
        ax = plt.gca()
        locs, labels = plt.yticks()  # 刻度的位置和它使用的数值
        plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))  # 每个位置上使用新的刻度值,可以将小数刻度变为指数刻度
        for tick in ax.xaxis.get_major_ticks():  # 获取图片坐标轴的主刻度,循环设置格式 解决指数坐标不能显示负号的问题
            tick.label1.set_fontproperties('stixgeneral')
        plt.xlabel('逆比例学习率指数')
        plt.ylabel('准确率')
        plt.title('不同学习率MLP神经网络平均准确率变化图(SGD-Invscaling-Power_t)')
        plt.ylim(0, max(max(MinACU), max(MeanACU)))
        plt.show()

    # 107、比较不同训练集比率
    def PlotTrainingSampleRatio(self,ACU_X, ACU, time, maxiter):
        fig, ax = plt.subplots()  # 传入的为训练集数量和对应的准确率
        ax.plot(ACU_X, ACU, 'g-v', linewidth=2, markersize=2, label='平均准确率')
        ax.plot([1, max(ACU_X)], [min(ACU), min(ACU)], 'r-o', label='平均准确率最小值', linewidth=1)
        ax.plot([1, max(ACU_X)], [np.mean(ACU), np.mean(ACU)], 'b-o', label='平均准确率平均值', linewidth=1)
        ax.fill_between(np.arange(1, max(ACU_X), 1),
                        np.mean(ACU) + np.std(ACU), np.mean(ACU) - np.std(ACU),
                        alpha=0.1, color='r')
        ax.text((1 + max(ACU_X)) / 2, np.mean(ACU) + 0.005, "准确率平均值 : " + str(round(np.mean(ACU), 5)),
                horizontalalignment='center', color='b', fontsize=16)
        ax.text((1 + max(ACU_X)) / 2, min(ACU) + 0.005, "准确率最小值 : " + str(round(min(ACU), 5)),
                horizontalalignment='center', color='r', fontsize=16)
        # ax.text((1 + max(ACU_X)) / 2, (min(ACU) + max(ACU)) / 2, "Std ACU : " + str(round(np.std(ACU), 5)),
        #         family="Times New Roman", fontsize=16)
        ax.legend(loc='upper right')
        ax.set_ylabel('平均准确率')
        ax.set_xlabel('训练样本数量/个')
        ax1 = ax.twinx()
        ax1.plot(ACU_X, time, 'c-d', linewidth=2, label='训练时间')
        ax1.set_ylabel('每次训练时间/s')
        ax1.legend()
        plt.title(f'MLP神经网络每{maxiter}次的平均准确率')
        plt.legend(loc='upper left')
        plt.show()

    '''7、比较svm和mlp不同训练样本数用到的函数'''
    # 108、svm模型固定maxiter次 某个训练比例下的概率 在获取不同训练及比例概率get_svm_acu_trainsize函数中调用
    def get_svm_acu_onetraining(self,clf, X, y, maxiter, trainsize=0.7):
        # 每一次划分训练集后应该迭代100次
        ACU = []
        for i in range(maxiter):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=trainsize)
            clf.fit(X_train, y_train)  # 训练模型
            y_pre = clf.predict(X_test)  # 某一次的预测结果
            true_num = 0
            for j in range(len(y_pre)):
                if y_pre[j] == y_test[j]:
                    true_num = true_num + 1
            acu = true_num / len(y_pre)  # 某一次的预测准确率
            ACU.append(acu)  # 100次的所有预测准确率
        return ACU

    # 109、mlp模型固定maxiter次 某个训练比例下的概率 在获取不同训练及比例概率get_mlp_acu_trainsize函数中调用
    def get_mlp_acu_onetraining(self,clf, X, y, train_size=0.7):
        # 一次训练得到的准确率 可以用来作为基础函数被反复调用
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        num_x_train = len(y_train)  # 返回训练样本数量
        true_num = 0  # 预测正确的数量
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                true_num = true_num + 1
        acu = true_num / len(y_pred)  # 单次预测的准确率
        return acu, num_x_train

    # 110、获得svm模型每个比例100次准确率的平均值、平均时间、该比例对应训练集个数
    def get_svm_acu_trainsize(self,clf, X, y, maxiter, trainsizes):
        Mean_ACU_Trainsize = []  # 不同训练比例下的100次平均准确率 11个值
        Mean_Time_Trainsize = []
        Train_num = []
        for trainsize in trainsizes:
            train_num = len(X) * trainsize
            A = time()
            ACU_Trainsize = self.get_svm_acu_transize(clf, X, y, maxiter, trainsize)  # 某一个训练比例下100次的准确率
            B = time()
            time_testsize = round(B - A, 3) / maxiter
            Mean_ACU = np.mean(ACU_Trainsize)  # 取100次的平均
            Mean_ACU_Trainsize.append(Mean_ACU)
            Mean_Time_Trainsize.append(time_testsize)
            Train_num.append(train_num)
        return Mean_ACU_Trainsize, Mean_Time_Trainsize, Train_num  # 每个比例100次准确率的平均值、平均时间、该比例对应训练集个数

    # 111、mlp在maxiter次改变训练集比例的准确率
    def get_mlp_acu_trainsize(self,clf, maxiter, train_size, X, y):
        # 获得某个transize下的准确率ACU 反复调用基础函数
        ACU = []
        num_X_train = []
        for i in range(maxiter):  # 训练代数改变
            acu, num_x_train = self.get_mlp_acu_onetraining(clf, X, y, train_size=train_size)  # 这是训练一次得到的准确率
            ACU.append(acu)
            num_X_train.append(num_x_train)
        return ACU, num_X_train

    # 112、改变MLP模型的训练集比例 调用get_mlp_acu_trainsize函数
    def ChangeMLPTrainsize(self,X, y):
        # 调用固定transize的maxiter次准确率---->调用基础函数求1次准确率
        trainsizes = np.arange(0.1, 1.0, 0.1)
        clf = MLPClassifier(activation='identity', solver='lbfgs', alpha=0.1, hidden_layer_sizes=(5, 2), random_state=1)
        maxiter = 100
        ACUmean = []
        ACU_X = []
        Time = []
        for trainsize in trainsizes:  # 不同训练样本数
            starttime = time()
            acu, acu_x = self.get_mlp_acu_trainsize(clf, maxiter, trainsize, X=X, y=y)  # 某一个训练比例,迭代maxiter次 得到相应的acu
            acu_mean = np.mean(acu)  # 找到每个训练比例下迭代maiter次的准确率平均值
            ACUmean.append(acu_mean)
            acu_x_mean = np.mean(acu_x)
            ACU_X.append(acu_x_mean)  # 每次训练的数量 这里平均都是一样的
            endtime = time()
            consumetime = endtime - starttime
            Time.append(consumetime)
        Time = np.array(Time) / maxiter  # 归算到每一次花费的时间
        self.plot_mlp_acu_trainsize(ACU_X, ACUmean, Time, maxiter=maxiter)

    # 113、改变SVM模型的训练集比例 调用get_svm_acu_trainsize函数
    def ChangeSVMTrainsize(self,X, y):
        testsizes = np.arange(0.1, 1.0, 0.1)
        clf = svm.SVC(kernel='linear', C=1, probability=True)
        maxiter = 100
        acu, time, acu_x = self.get_svm_acu_trainsize(clf, X, y, maxiter, testsizes)  # 导入模型、迭代次数、训练比例
        self.plot_svm_acu_trainsize(acu_x, acu, time, maxiter)

    # 114、挥着svm模型随训练比例的准确率变化
    def plot_svm_acu_trainsize(self,ACU_X, ACU, time, maxiter):
        fig, ax = plt.subplots()
        ax.plot(ACU_X, ACU, 'g-v', linewidth=2, markersize=2, label='平均准确率')
        ax.plot([1, max(ACU_X)], [min(ACU), min(ACU)], 'r-o', label='平均准确率最小值', linewidth=1)
        ax.plot([1, max(ACU_X)], [np.mean(ACU), np.mean(ACU)], 'b-o', label='平均准确率平均值', linewidth=1)

        ax.fill_between(np.arange(1, max(ACU_X), 1), np.mean(ACU) + np.std(ACU), np.mean(ACU) - np.std(ACU), alpha=0.1,
                        color='r')
        ax.text((1 + max(ACU_X)) / 2, np.mean(ACU) + 0.005, "准确率平均值 : " + str(round(np.mean(ACU), 5)),
                horizontalalignment='center', color='b', fontsize=16)
        ax.text((1 + max(ACU_X)) / 2, min(ACU) + 0.005, "准确率最小值 : " + str(round(min(ACU), 5)),
                horizontalalignment='center', color='r', fontsize=16)
        ax.legend(loc='upper right')
        ax.set_ylabel('平均准确率')
        ax.set_xlabel('训练样本数量/个')
        ax1 = ax.twinx()
        ax1.plot(ACU_X, time, 'c-d', linewidth=2, label='训练时间')
        ax1.set_ylabel('每次训练时间/s')
        ax1.legend()
        plt.title(f'支持向量机SVM每{maxiter}次的平均准确率')
        plt.legend(loc='upper left')
        plt.show()

    # 115、绘制mlp模型随训练比例的准确率变化
    def plot_mlp_acu_trainsize(self,ACU_X, ACU, time, maxiter):
        fig, ax = plt.subplots()
        ax.plot(ACU_X, ACU, 'g-v', linewidth=2, markersize=2, label='平均准确率')  # 折线图
        ax.plot([1, max(ACU_X)], [min(ACU), min(ACU)], 'r-o', label='平均准确率最小值', linewidth=1)  # 直线
        ax.plot([1, max(ACU_X)], [np.mean(ACU), np.mean(ACU)], 'b-o', label='平均准确率平均值', linewidth=1)  # 直线

        ax.fill_between(np.arange(1, max(ACU_X), 1),
                        np.mean(ACU) + np.std(ACU), np.mean(ACU) - np.std(ACU),
                        alpha=0.1, color='r')
        ax.text((1 + max(ACU_X)) / 2, np.mean(ACU) + 0.005,
                "准确率平均值 : " + str(round(np.mean(ACU), 5)),
                horizontalalignment='center', color='b', fontsize=16)
        ax.text((1 + max(ACU_X)) / 2, min(ACU) + 0.005,
                "准确率最小值 : " + str(round(min(ACU), 5)),
                horizontalalignment='center', color='r', fontsize=16)
        # ax.text((1 + max(ACU_X)) / 2, (min(ACU) + max(ACU)) / 2, "Std ACU : " + str(round(np.std(ACU), 5)),
        #         family="Times New Roman", fontsize=16)
        ax.legend(loc='upper right')
        ax.set_ylabel('平均准确率')
        ax.set_xlabel('训练样本数量/个')
        ax1 = ax.twinx()
        ax1.plot(ACU_X, time, 'c-d', linewidth=2, label='训练时间')
        ax1.set_ylabel('每次训练时间/s')
        ax1.legend()
        plt.title(f'MLP神经网络每{maxiter}次的平均准确率')
        plt.legend(loc='upper left')
        plt.show()

    '''8、比较svm和mlp不同训练次数用到的函数'''
    # 116、svm模型改变训练次数(固定训练集比例)
    def get_svm_acu_trainsize_maxiter(self,clf, X, y, maxiter, trainsize=0.7):
        # 每一次划分训练集后应该迭代100次
        PP = []
        for i in range(maxiter):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=trainsize)
            clf.fit(X_train, y_train)  # 训练模型
            y_pre = clf.predict(X_test)  # 某一次的预测结果
            pp = 0
            for j in range(len(y_pre)):
                if y_pre[j] == y_test[j]:
                    pp = pp + 1
            pp = pp / len(y_pre)  # 某一次的预测准确率
            PP.append(pp)  # 100次的所有预测准确率
        return PP, maxiter, trainsize

    # 117、mlp模型改变训练次数(固定训练集比例)
    def get_mlp_acu_trainsize_maxiter(self,clf, maxiter,  X, y,train_size=0.7):
        # 获得某个transize下的准确率ACU 反复调用基础函数
        ACU = []
        num_X_train = []
        for i in range(maxiter):  # 训练代数改变
            acu, num_x_train = self.get_mlp_acu_onetraining(clf, X, y, train_size=train_size)  # 这是训练一次得到的准确率
            ACU.append(acu)
            num_X_train.append(num_x_train)
        return ACU, num_X_train

    # 118、绘制svm不同训练次数准确率变化
    def plot_svm_acu_maxiter(self,ACU, maxiter, trainsize):
        plt.plot(ACU, 'c-p', linewidth=1, markersize=2, label='准确率')
        plt.plot([1, maxiter], [min(ACU), min(ACU)], 'r-o', label='准确率最小值', linewidth=1)
        plt.plot([1, maxiter], [np.mean(ACU), np.mean(ACU)], 'b-o', label='准确率平均值', linewidth=1)
        plt.fill_between(np.arange(1, maxiter, 1), np.mean(ACU) + np.std(ACU), np.mean(ACU) - np.std(ACU), alpha=0.1,
                         color='r')
        plt.text((1 + maxiter) / 2, np.mean(ACU) + 0.005, "准确率平均值 : " + str(round(np.mean(ACU), 5))
                 , horizontalalignment='center', color='b', fontsize=16)
        plt.text((1 + maxiter) / 2, min(ACU) + 0.005, "准确率最小值 : " + str(round(min(ACU), 5)),
                 horizontalalignment='center', color='r', fontsize=16)
        plt.title(f'支持向量机SVM准确率变化图(训练集比例:{trainsize})')
        plt.ylabel('准确率')
        plt.xlabel('训练次数')
        plt.legend(loc='lower left')
        plt.show()

    # 119、绘制mlp不同训练次数准确率变化
    def plot_mlp_acu_maxiter(ACU, maxiter, trainsize):
        plt.plot(ACU, 'c-p', linewidth=1, markersize=2, label='准确率')
        plt.plot([1, maxiter], [min(ACU), min(ACU)], 'r-o', label='准确率最小值', linewidth=1)
        plt.plot([1, maxiter], [np.mean(ACU), np.mean(ACU)], 'b-o', label='准确率平均值', linewidth=1)
        plt.fill_between(np.arange(1, maxiter, 1),
                         np.mean(ACU) + np.std(ACU), np.mean(ACU) - np.std(ACU),
                         alpha=0.1, color='r')
        plt.text((1 + maxiter) / 2, np.mean(ACU) + 0.005,
                 "准确率平均值 : " + str(round(np.mean(ACU), 5))
                 , horizontalalignment='center', color='b', fontsize=16)
        plt.text((1 + maxiter) / 2, min(ACU) + 0.005,
                 "准确率最小值 : " + str(round(min(ACU), 5)),
                 horizontalalignment='center', color='r', fontsize=16)
        plt.title(f'MLP神经网络准确率变化图(训练集比例:{trainsize})')
        plt.ylabel('准确率')
        plt.xlabel('训练次数')
        plt.legend(loc='lower left')
        plt.show()

    '''9、比较留出法'''
    # 120、一次训练
    def one_training(self,clf, X, y, testsize):
        # 一次训练得到的准确率 可以用来作为基础函数被反复调用
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testsize)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        num_x_train = len(y_train)  # 返回训练样本数量
        true_num = 0  # 预测正确的数量
        for i in range(len(y_pred)):
            if y_pred[i] == y_test[i]:
                true_num = true_num + 1
        acu = true_num / len(y_pred)  # 准确率
        return acu, num_x_train

    # 121、maxiter次训练
    def maxiter_training(self,clf, X, y, maxiter, testsize):
        # 获得某个transize下的准确率ACU，作为基础函数反复调用，也调用基础函数one_training
        ACU = []
        num_X_train = []  # 固定样本数时此参数忽略
        for i in range(maxiter):  # 训练代数改变 如100次训练的准确率 需要ACU存储
            acu, num_x_train = self.one_training(clf, X, y, testsize=testsize)  # 这是训练一次得到的准确率
            ACU.append(acu)
            num_X_train.append(num_x_train)
        return ACU, num_X_train

    # 122、固定测试集比例
    def fixed_testsize(self,kernel, title, X, y):
        maxiter = 100
        testsize = 0.3
        clf = svm.SVC(kernel=kernel, C=1, probability=True)
        # oneacu = onetrain(clf,X,y,testsize) # 训练一次的准确率是不够的
        ACU, _ = self.maxiter_training(clf, X=X, y=y, maxiter=maxiter, testsize=testsize)  # 固定样本数时此参数忽略
        self.plot_acu_fixed_testsize(ACU, testsize, title, maxiter=maxiter)

    # 123、固定训练次数
    def fixed_training_num(self,kernel, title, X, y):
        clf = svm.SVC(kernel=kernel, C=1, probability=True)
        testsizes = np.arange(0.1, 1.0, 0.1)
        maxiter = 100
        ACUmean = []
        ACU_X = []
        Time = []
        for testsize in testsizes:  # 不同训练样本数
            starttime = time()
            acu, acu_x = self.maxiter_training(clf, X, y, maxiter, testsize)  # 某一个训练比例,迭代maxiter次 得到相应的acu 和对应的样本数
            acu_mean = np.mean(acu)  # 找到每个训练比例下迭代maiter次的准确率平均值 这是因为固定训练次数的话需要1个比例训练多次所以要取平均
            ACUmean.append(acu_mean)
            acu_x_mean = np.mean(acu_x)
            ACU_X.append(acu_x_mean)  # 每次训练的数量
            endtime = time()
            consumetime = endtime - starttime
            Time.append(consumetime)
        Time = np.array(Time) / maxiter  # 归算到每一次花费的时间
        self.plot_acu_fixed_training_num(ACU_X, ACUmean, Time, title, maxiter)

    # 124、绘制固定测试集比例的acu
    def plot_acu_fixed_testsize(self,ACU, testsize, title, maxiter):
        # 绘制固定测试集比例 不同训练次数时的准确率
        plt.plot(ACU, 'c-p', linewidth=1, markersize=2, label='准确率')
        plt.plot([1, maxiter], [min(ACU), min(ACU)], 'r-o', label='准确率最小值', linewidth=1, )
        plt.plot([1, maxiter], [np.mean(ACU), np.mean(ACU)], 'b-o', label='准确率平均值', linewidth=1)
        plt.fill_between(np.arange(1, maxiter, 1), np.mean(ACU) + np.std(ACU), np.mean(ACU) - np.std(ACU), alpha=0.1,
                         color='r')
        plt.text((1 + maxiter) / 2, np.mean(ACU) + 0.01,
                 "准确率平均值 : " + str(round(np.mean(ACU), 5)),
                 horizontalalignment='center', fontsize=16, color='r')
        plt.text((1 + maxiter) / 2, min(ACU) + 0.01,
                 "准确率最小值 : " + str(round(min(ACU), 5)),
                 horizontalalignment='center', fontsize=16, color='b')
        plt.text((1 + maxiter) / 2, (min(ACU) + max(ACU)) / 2 - 0.04,
                 "准确率标准差 : : " + str(round(np.std(ACU), 5)),
                 horizontalalignment='center', fontsize=16)
        plt.title(title + f'准确率变化图(测试集比例:{testsize})')
        plt.ylabel('准确率')
        plt.xlabel('训练次数')
        plt.legend(loc='lower left')
        plt.show()

    # 125、绘制固定训练次数的acu
    def plot_acu_fixed_training_num(self,ACU_X, ACU, time, title, maxiter):
        '''
         绘制固定训练次数时 不同测试集比例的准确率
        :param ACU_X: 样本数量 list
        :param ACU: 不同样本数量对应的maxiter次平均准确率
        :param time: 不同样本数量对应的maxiter次 时间
        :return: 预测准确率和消耗时间 趋势图
        '''
        fig, ax = plt.subplots()
        ax.plot(ACU_X, ACU, 'g-v', linewidth=2, markersize=2, label='平均准确率')
        ax.plot([1, max(ACU_X)], [min(ACU), min(ACU)], 'r-o', label='平均正确率最小值', linewidth=1)
        ax.plot([1, max(ACU_X)], [np.mean(ACU), np.mean(ACU)], 'b-o', label='平均正确率平均值', linewidth=1)
        ax.fill_between(np.arange(1, max(ACU_X), 1), np.mean(ACU) + np.std(ACU), np.mean(ACU) - np.std(ACU), alpha=0.1,
                        color='r')
        ax.text((1 + max(ACU_X)) / 2, np.mean(ACU) + 0.02,
                "准确率平均值 : " + str(round(np.mean(ACU), 5)),
                horizontalalignment='center', family="Times New Roman", fontsize=16)
        ax.text((1 + max(ACU_X)) / 2, min(ACU) + 0.02,
                "准确率最小值 : " + str(round(min(ACU), 5)),
                horizontalalignment='center', fontsize=16)
        ax.text((1 + max(ACU_X)) / 2, (min(ACU) + max(ACU)) / 2,
                "准确率标准差 : " + str(round(np.std(ACU), 5)),
                horizontalalignment='center', fontsize=16)
        ax.legend(loc='upper right')
        ax.set_ylabel('平均准确率')
        ax.set_xlabel('训练样本数量/个')
        ax1 = ax.twinx()
        ax1.plot(ACU_X, time, 'c-d', linewidth=2, label='训练时间')
        ax1.set_ylabel('每次训练时间/s')
        ax1.legend()
        plt.title(title + f'每{maxiter}次的平均准确率')
        plt.legend(loc='upper left')
        plt.show()

    '''10、比较留1法和留P法'''
    # 126、留1法或留2法
    def return_leave_p_method_acu_training_num(self,clf, X, y, loo, p):
        '''
        :param clf: 传入的模型
        :param X: 训练数据
        :param y: 训练标签
        :param loo: 划分方法 loo = LeaveOneOut() or  LeavePOut(p=p)
        :param p: 1 or 2
        :return: 当前训练次数和当前的准确率变化
        '''
        # loo根据p的取值得到的划分方法
        training_nums = 0  # 用来计算当前的训练次数，留1法最终240，留2法就需要计算C(n,2)
        num = 0  # 计算留1法到当前训练次数预测正确的个数
        sumacu = 0  # 留2法计算当前训练次数总的准确率
        ACU = []
        for train_index, test_index in loo.split(X):
            X_train, y_train = X[train_index], y[train_index]
            X_test, y_test = X[test_index], y[test_index]
            idx = loo.get_n_splits(X)  # n_splits = 241
            pp = 0  # 用于留2法确定每次预测时2个有几个正确 取值只有0,0.5,1.0
            if idx:  # 每次划分都会返回一个bool值 为真则记录一次数据K 最终K=240 或者说训练集数-1
                training_nums = training_nums + 1
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            if p == 1:
                if y_pred == y_test:  # 每次交叉验证只预测1个测试集,故直接进行比较即可
                    num = num + 1
                acu = num / training_nums  # 准确率 (判断当前240交叉验证次数下有多少个预测正确)
            else:  # 留2法时还需要计算每次2个中有几个正确,取值只有0,0.5,1.0
                for j in range(len(y_pred)):
                    if y_pred[j] == y_test[j]:
                        pp = pp + 1
                acu = pp / len(y_pred)  # 0,0.5,1.0
                sumacu = sumacu + acu  # 到当前k次训练时的准确率总和
                acu = sumacu / training_nums  # 当前准确率总和/当前训练总次数 = 当前准确率 会随着训练次数持续变化
            ACU.append(acu)
        return ACU, training_nums  # 返回每次训练的当前准确率和最终训练次数

    # 127、可视化留1法或留2法的准确率变化趋势
    def plot_leave_p_method_acu(self,ACU, title, TrainingNums, Time):
        plt.plot(ACU, 'c-p', linewidth=2, markersize=2, label='准确率')
        plt.plot([1, TrainingNums], [min(ACU), min(ACU)], 'r-o', label='准确率最小值', linewidth=1)
        plt.plot([1, TrainingNums], [np.mean(ACU), np.mean(ACU)], 'b-o', label='准确率平均值', linewidth=1)
        plt.fill_between(np.arange(1, TrainingNums, 1), np.mean(ACU) + np.std(ACU), np.mean(ACU) - np.std(ACU),
                         alpha=0.1,
                         color='r')
        plt.text((1 + TrainingNums) / 2, np.mean(ACU) + 0.008, "Avarage ACU : " + str(round(np.mean(ACU), 5)),
                 family="Times New Roman", horizontalalignment='center', fontsize=16)
        plt.text((1 + TrainingNums) / 2, min(ACU) + 0.005, "Min ACU : " + str(round(min(ACU), 5)),
                 family="Times New Roman",
                 fontsize=16, horizontalalignment='center')
        plt.text((1 + TrainingNums) / 2, (min(ACU) + max(ACU)) / 2, "Std ACU : " + str(round(np.std(ACU), 5)),
                 family="Times New Roman", horizontalalignment='center', fontsize=16)
        plt.text((1 + TrainingNums) / 2, min(ACU) + 0.008,
                 "Training Time : " + str(round(Time / TrainingNums, 3)) + ' s',
                 verticalalignment='center', horizontalalignment='center', family='Times New Roman', fontsize=16)
        plt.title(title + f'准确率变化图')
        plt.ylabel('准确率')
        plt.xlabel('训练次数')
        plt.legend(loc='lower left')
        plt.show()

    # 128、留P法
    def leave_p_method(self,kernel, title, X, y, p):
        '''
        :param kernel: 核函数类型
        :param title: 图标题
        :param X: 数据集
        :param y: 标签集
        :param p: 留p法 p=1、2
        :return: 留1或留2法的准确率到当前次数的变化曲线
        '''
        if p == 1:
            loo = LeaveOneOut()
        else:
            loo = LeavePOut(p=p)
        clf = svm.SVC(kernel=kernel, C=1, probability=True)
        A = time()
        acu, TrainingNums = self.return_leave_p_method_acu_training_num(clf, X, y, loo, p)  # 随着交叉验证/训练次数的变化而变化的准确率
        B = time()
        Time = B - A
        self.plot_leave_p_method_acu(acu, title, TrainingNums, Time)

    '''11、比较不同核函数的pca数据边界'''
    # 129、绘制曲面
    def make_meshgrid(self,x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1  # 得到最小值和最大值加减1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),  # 得到二维网格图
                             np.arange(y_min, y_max, h))
        return xx, yy

    # 130、热力图
    def plot_contours(self,ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out, Z

    # 131、绘制不同核函数的pca数据边界
    def plot_kernel_contours(self,X, y):
        '''
        :param X: 数据
        :param y: 数据标签
        :return: 比较不同核函数的pca数据边界
        '''
        C = 0.5  # 正则化系数,值越小泛化能力强,但是容易欠拟合,反之容易过拟合
        tol = 0.0001  # 迭代阈值
        models = (svm.SVC(kernel='linear', C=C, tol=tol),
                  svm.SVC(kernel='rbf', gamma=0.1, C=C, tol=tol),
                  svm.SVC(kernel='poly', degree=3, gamma='auto', C=C, tol=tol),  # 'auto' gamma= 1 / n_features.
                  svm.SVC(kernel='sigmoid', C=C, tol=tol))
        models = (clf.fit(X, y) for clf in models)
        titles = ('线性核函数',
                  '径向基核函数',
                  '多项式核函数(3阶)',
                  'Sigmoid核函数')
        fig, sub = plt.subplots(2, 2)
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        X0, X1 = X[:, 0], X[:, 1]  # 分别是第一列和第二列
        xx, yy = self.make_meshgrid(X0, X1)
        for clf, title, ax in zip(models, titles, sub.flatten()):
            Z = self.plot_contours(ax, clf, xx, yy,
                              cmap=plt.cm.hsv, alpha=0.8)
            ax.scatter(X0, X1, c=y, cmap=plt.cm.hsv, s=40, edgecolors='k')
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xlabel('PCA2')
            ax.set_ylabel('PCA1')
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(title)
        plt.show()  # 不分训练和测试集的不同核函数的横向对比

app = QApplication(sys.argv)
my_platform = myPlatform()
# my_platform .setWindowIcon(QIcon(":/images/platformIcon.ico"))
sys.exit(app.exec_())
