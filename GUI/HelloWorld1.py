# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'NormalDialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.
'''
一个完整的工程，带有槽函数和信号关联的示例工程
'''

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_NormalDialog(object):
    def setupUi(self, NormalDialog):
        '''设置窗体,继承了Dialog类'''
        NormalDialog.setObjectName("NormalDialog")
        NormalDialog.resize(1000, 800)
        self.layoutWidget = QtWidgets.QWidget(NormalDialog)
        self.layoutWidget.setGeometry(QtCore.QRect(100, 100, 800, 600))
        self.layoutWidget.setObjectName("layoutWidget")
        '''水平布局的名称叫verticalLayout,设置到上下左右边缘的距离'''
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.StyleBox = QtWidgets.QGroupBox(self.layoutWidget)
        self.StyleBox.setTitle("")
        self.StyleBox.setObjectName("StyleBox")
        self.StyleLayout = QtWidgets.QHBoxLayout(self.StyleBox)
        self.StyleLayout.setContentsMargins(-1, -1, -1, 9)
        self.StyleLayout.setSpacing(6)
        self.StyleLayout.setObjectName("StyleLayout")
        '''依次设置3个box的名称、边缘距离和布局'''
        self.Bold = QtWidgets.QCheckBox(self.StyleBox)
        self.Bold.setChecked(True)
        self.Bold.setObjectName("Bold")
        self.StyleLayout.addWidget(self.Bold)
        self.Underline = QtWidgets.QCheckBox(self.StyleBox)
        self.Underline.setChecked(False)
        self.Underline.setObjectName("Underline")
        self.StyleLayout.addWidget(self.Underline)
        self.Italic = QtWidgets.QCheckBox(self.StyleBox)
        self.Italic.setChecked(False)
        self.Italic.setAutoRepeat(False)
        self.Italic.setObjectName("Italic")
        self.StyleLayout.addWidget(self.Italic)
        self.verticalLayout.addWidget(self.StyleBox)
        self.ColorBox = QtWidgets.QGroupBox(self.layoutWidget)
        self.ColorBox.setTitle("")
        self.ColorBox.setObjectName("ColorBox")
        self.ColorLayout = QtWidgets.QHBoxLayout(self.ColorBox)
        self.ColorLayout.setObjectName("ColorLayout")
        self.Red = QtWidgets.QRadioButton(self.ColorBox)
        self.Red.setChecked(True)
        self.Red.setObjectName("Red")
        self.ColorLayout.addWidget(self.Red)
        self.Blue = QtWidgets.QRadioButton(self.ColorBox)
        self.Blue.setChecked(False)
        self.Blue.setObjectName("Blue")
        self.ColorLayout.addWidget(self.Blue)
        self.Black = QtWidgets.QRadioButton(self.ColorBox)
        self.Black.setObjectName("Black")
        self.ColorLayout.addWidget(self.Black)
        self.verticalLayout.addWidget(self.ColorBox)
        '''设置文本格式'''
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.layoutWidget)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.verticalLayout.addWidget(self.plainTextEdit)
        '''设置按钮box的格式'''
        self.ButtonBox = QtWidgets.QGroupBox(self.layoutWidget)
        self.ButtonBox.setTitle("")
        self.ButtonBox.setObjectName("ButtonBox")
        self.ButtonLayout = QtWidgets.QHBoxLayout(self.ButtonBox)
        self.ButtonLayout.setObjectName("ButtonLayout")
        self.Sure = QtWidgets.QPushButton(self.ButtonBox)
        self.Sure.setObjectName("Sure")
        self.ButtonLayout.addWidget(self.Sure)
        self.DropOut = QtWidgets.QPushButton(self.ButtonBox)
        self.DropOut.setObjectName("DropOut")
        self.ButtonLayout.addWidget(self.DropOut)
        self.Clear = QtWidgets.QPushButton(self.ButtonBox)
        self.Clear.setObjectName("Clear")
        self.ButtonLayout.addWidget(self.Clear)
        self.verticalLayout.addWidget(self.ButtonBox)

        self.retranslateUi(NormalDialog)
        '''此三句程序表示信号和槽的关联'''
        self.Sure.clicked.connect(NormalDialog.accept)
        self.DropOut.clicked.connect(NormalDialog.close)
        QtCore.QMetaObject.connectSlotsByName(NormalDialog)

    def retranslateUi(self, NormalDialog):
        _translate = QtCore.QCoreApplication.translate
        NormalDialog.setWindowTitle(_translate("NormalDialog", "NormalDialog"))
        self.Bold.setText(_translate("NormalDialog", "Bold"))
        self.Underline.setText(_translate("NormalDialog", "Underline"))
        self.Italic.setText(_translate("NormalDialog", "Italic"))
        self.Red.setText(_translate("NormalDialog", "Red"))
        self.Blue.setText(_translate("NormalDialog", "Blue"))
        self.Black.setText(_translate("NormalDialog", "Black"))
        self.plainTextEdit.setPlainText(_translate("NormalDialog", "Hello World!\n"
"My name is chenbei.\n"
"My hobby is basketball.\n"
"Python is the best language in the world!\n"
"I don\'t have girlfriend,maybe there will be a miracle in the near future."))
        self.Sure.setText(_translate("NormalDialog", "确定"))
        self.DropOut.setText(_translate("NormalDialog", "退出"))
        self.Clear.setText(_translate("NormalDialog", "清空"))
