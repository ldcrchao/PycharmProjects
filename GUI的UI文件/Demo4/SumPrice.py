# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SumPrice.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_SumPrice(object):
    def setupUi(self, SumPrice):
        SumPrice.setObjectName("SumPrice")
        SumPrice.resize(1000, 800)
        self.widget = QtWidgets.QWidget(SumPrice)
        self.widget.setGeometry(QtCore.QRect(100, 100, 800, 600))
        self.widget.setObjectName("widget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.widget)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.groupBox = QtWidgets.QGroupBox(self.widget)
        self.groupBox.setObjectName("groupBox")
        self.widget1 = QtWidgets.QWidget(self.groupBox)
        self.widget1.setGeometry(QtCore.QRect(0, 20, 421, 131))
        self.widget1.setObjectName("widget1")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.widget1)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.numlabel = QtWidgets.QLabel(self.widget1)
        self.numlabel.setObjectName("numlabel")
        self.horizontalLayout.addWidget(self.numlabel)
        self.num = QtWidgets.QLineEdit(self.widget1)
        self.num.setObjectName("num")
        self.horizontalLayout.addWidget(self.num)
        self.pricelabel = QtWidgets.QLabel(self.widget1)
        self.pricelabel.setObjectName("pricelabel")
        self.horizontalLayout.addWidget(self.pricelabel)
        self.price = QtWidgets.QLineEdit(self.widget1)
        self.price.setObjectName("price")
        self.horizontalLayout.addWidget(self.price)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.CalculateButton = QtWidgets.QPushButton(self.widget1)
        self.CalculateButton.setObjectName("CalculateButton")
        self.horizontalLayout_3.addWidget(self.CalculateButton)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.sumpricelabel = QtWidgets.QLabel(self.widget1)
        self.sumpricelabel.setObjectName("sumpricelabel")
        self.horizontalLayout_2.addWidget(self.sumpricelabel)
        self.sumprice = QtWidgets.QLineEdit(self.widget1)
        self.sumprice.setObjectName("sumprice")
        self.horizontalLayout_2.addWidget(self.sumprice)
        self.horizontalLayout_3.addLayout(self.horizontalLayout_2)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.verticalLayout_3.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.widget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.widget2 = QtWidgets.QWidget(self.groupBox_2)
        self.widget2.setGeometry(QtCore.QRect(0, 20, 421, 131))
        self.widget2.setObjectName("widget2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.widget2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.numSpinBox = QtWidgets.QLabel(self.widget2)
        self.numSpinBox.setObjectName("numSpinBox")
        self.horizontalLayout_4.addWidget(self.numSpinBox)
        self.numspinBox = QtWidgets.QSpinBox(self.widget2)
        self.numspinBox.setPrefix("")
        self.numspinBox.setMaximum(100)
        self.numspinBox.setSingleStep(0)
        self.numspinBox.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.numspinBox.setProperty("value", 10)
        self.numspinBox.setObjectName("numspinBox")
        self.horizontalLayout_4.addWidget(self.numspinBox)
        self.priceSpinBox = QtWidgets.QLabel(self.widget2)
        self.priceSpinBox.setObjectName("priceSpinBox")
        self.horizontalLayout_4.addWidget(self.priceSpinBox)
        self.pricedoubleSpinBox = QtWidgets.QDoubleSpinBox(self.widget2)
        self.pricedoubleSpinBox.setMinimum(50.0)
        self.pricedoubleSpinBox.setMaximum(1000.0)
        self.pricedoubleSpinBox.setObjectName("pricedoubleSpinBox")
        self.horizontalLayout_4.addWidget(self.pricedoubleSpinBox)
        self.verticalLayout_2.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.autocalculateSpinBox = QtWidgets.QLabel(self.widget2)
        self.autocalculateSpinBox.setObjectName("autocalculateSpinBox")
        self.horizontalLayout_5.addWidget(self.autocalculateSpinBox)
        self.autocalpricedoubleSpinBox = QtWidgets.QDoubleSpinBox(self.widget2)
        self.autocalpricedoubleSpinBox.setMinimum(50.0)
        self.autocalpricedoubleSpinBox.setMaximum(1000.0)
        self.autocalpricedoubleSpinBox.setObjectName("autocalpricedoubleSpinBox")
        self.horizontalLayout_5.addWidget(self.autocalpricedoubleSpinBox)
        self.verticalLayout_2.addLayout(self.horizontalLayout_5)
        self.verticalLayout_3.addWidget(self.groupBox_2)

        self.retranslateUi(SumPrice)
        QtCore.QMetaObject.connectSlotsByName(SumPrice)

    def retranslateUi(self, SumPrice):
        _translate = QtCore.QCoreApplication.translate
        SumPrice.setWindowTitle(_translate("SumPrice", "SumPrice"))
        self.groupBox.setTitle(_translate("SumPrice", "QLineEdit输入和显示数值"))
        self.numlabel.setText(_translate("SumPrice", "数量"))
        self.pricelabel.setText(_translate("SumPrice", "单价"))
        self.CalculateButton.setText(_translate("SumPrice", "点击计算"))
        self.sumpricelabel.setText(_translate("SumPrice", "总价"))
        self.groupBox_2.setTitle(_translate("SumPrice", "SpinBox输入和显示数值"))
        self.numSpinBox.setText(_translate("SumPrice", "数量"))
        self.numspinBox.setSuffix(_translate("SumPrice", " kg "))
        self.priceSpinBox.setText(_translate("SumPrice", "单价"))
        self.pricedoubleSpinBox.setPrefix(_translate("SumPrice", "$ "))
        self.autocalculateSpinBox.setText(_translate("SumPrice", "自动计算总价"))
        self.autocalpricedoubleSpinBox.setPrefix(_translate("SumPrice", "$ "))
