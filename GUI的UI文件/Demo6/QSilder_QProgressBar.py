# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'QSilder_QProgressBar.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_QSilder_QProgressBar(object):
    def setupUi(self, QSilder_QProgressBar):
        QSilder_QProgressBar.setObjectName("QSilder_QProgressBar")
        QSilder_QProgressBar.resize(800, 600)
        self.widget = QtWidgets.QWidget(QSilder_QProgressBar)
        self.widget.setGeometry(QtCore.QRect(40, 10, 481, 361))
        self.widget.setObjectName("widget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.SetProgressBar_2 = QtWidgets.QGroupBox(self.widget)
        self.SetProgressBar_2.setMinimumSize(QtCore.QSize(150, 70))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(15)
        self.SetProgressBar_2.setFont(font)
        self.SetProgressBar_2.setObjectName("SetProgressBar_2")
        self.verticalLayout_6 = QtWidgets.QVBoxLayout(self.SetProgressBar_2)
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.ScrollBarlabel = QtWidgets.QLabel(self.SetProgressBar_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(15)
        self.ScrollBarlabel.setFont(font)
        self.ScrollBarlabel.setObjectName("ScrollBarlabel")
        self.horizontalLayout_2.addWidget(self.ScrollBarlabel)
        self.horizontalScrollBar = QtWidgets.QScrollBar(self.SetProgressBar_2)
        self.horizontalScrollBar.setMinimumSize(QtCore.QSize(100, 20))
        self.horizontalScrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalScrollBar.setObjectName("horizontalScrollBar")
        self.horizontalLayout_2.addWidget(self.horizontalScrollBar)
        self.verticalLayout_6.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.Sliderlabel = QtWidgets.QLabel(self.SetProgressBar_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(15)
        self.Sliderlabel.setFont(font)
        self.Sliderlabel.setObjectName("Sliderlabel")
        self.horizontalLayout.addWidget(self.Sliderlabel)
        self.horizontalSlider = QtWidgets.QSlider(self.SetProgressBar_2)
        self.horizontalSlider.setMinimumSize(QtCore.QSize(100, 20))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        self.horizontalLayout.addWidget(self.horizontalSlider)
        self.verticalLayout_6.addLayout(self.horizontalLayout)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.ProgresslBarlabel = QtWidgets.QLabel(self.SetProgressBar_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(15)
        self.ProgresslBarlabel.setFont(font)
        self.ProgresslBarlabel.setObjectName("ProgresslBarlabel")
        self.horizontalLayout_3.addWidget(self.ProgresslBarlabel)
        self.progressBar = QtWidgets.QProgressBar(self.SetProgressBar_2)
        self.progressBar.setMinimumSize(QtCore.QSize(100, 20))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        self.horizontalLayout_3.addWidget(self.progressBar)
        self.verticalLayout_6.addLayout(self.horizontalLayout_3)
        self.verticalLayout.addWidget(self.SetProgressBar_2)
        self.SetProgressBar = QtWidgets.QGroupBox(self.widget)
        self.SetProgressBar.setMinimumSize(QtCore.QSize(150, 70))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(15)
        self.SetProgressBar.setFont(font)
        self.SetProgressBar.setObjectName("SetProgressBar")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.SetProgressBar)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.textVisible = QtWidgets.QCheckBox(self.SetProgressBar)
        self.textVisible.setMinimumSize(QtCore.QSize(30, 15))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.textVisible.setFont(font)
        self.textVisible.setObjectName("textVisible")
        self.horizontalLayout_4.addWidget(self.textVisible)
        self.InvertedAppearance = QtWidgets.QCheckBox(self.SetProgressBar)
        self.InvertedAppearance.setMinimumSize(QtCore.QSize(60, 15))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.InvertedAppearance.setFont(font)
        self.InvertedAppearance.setObjectName("InvertedAppearance")
        self.horizontalLayout_4.addWidget(self.InvertedAppearance)
        self.verticalLayout_4.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.recentvalue = QtWidgets.QRadioButton(self.SetProgressBar)
        self.recentvalue.setMinimumSize(QtCore.QSize(60, 15))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.recentvalue.setFont(font)
        self.recentvalue.setObjectName("recentvalue")
        self.horizontalLayout_5.addWidget(self.recentvalue)
        self.percent = QtWidgets.QRadioButton(self.SetProgressBar)
        self.percent.setMinimumSize(QtCore.QSize(60, 15))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.percent.setFont(font)
        self.percent.setObjectName("percent")
        self.horizontalLayout_5.addWidget(self.percent)
        self.verticalLayout_4.addLayout(self.horizontalLayout_5)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.Sure = QtWidgets.QPushButton(self.SetProgressBar)
        self.Sure.setMinimumSize(QtCore.QSize(30, 15))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(16)
        self.Sure.setFont(font)
        self.Sure.setObjectName("Sure")
        self.verticalLayout_2.addWidget(self.Sure)
        self.verticalLayout_4.addLayout(self.verticalLayout_2)
        self.verticalLayout_5.addLayout(self.verticalLayout_4)
        self.verticalLayout.addWidget(self.SetProgressBar)

        self.retranslateUi(QSilder_QProgressBar)
        self.Sure.clicked.connect(QSilder_QProgressBar.close)
        QtCore.QMetaObject.connectSlotsByName(QSilder_QProgressBar)

    def retranslateUi(self, QSilder_QProgressBar):
        _translate = QtCore.QCoreApplication.translate
        QSilder_QProgressBar.setWindowTitle(_translate("QSilder_QProgressBar", "QSilder_QProgressBar"))
        self.SetProgressBar_2.setTitle(_translate("QSilder_QProgressBar", "滑动条设置"))
        self.ScrollBarlabel.setText(_translate("QSilder_QProgressBar", "ScrollBar"))
        self.Sliderlabel.setText(_translate("QSilder_QProgressBar", "Slider"))
        self.ProgresslBarlabel.setText(_translate("QSilder_QProgressBar", "ProgresslBar"))
        self.SetProgressBar.setTitle(_translate("QSilder_QProgressBar", "进度条设置"))
        self.textVisible.setText(_translate("QSilder_QProgressBar", "textVisible"))
        self.InvertedAppearance.setText(_translate("QSilder_QProgressBar", "InvertedAppearance"))
        self.recentvalue.setText(_translate("QSilder_QProgressBar", "显示格式-当前值"))
        self.percent.setText(_translate("QSilder_QProgressBar", "显示格式-百分比"))
        self.Sure.setText(_translate("QSilder_QProgressBar", "确定"))
