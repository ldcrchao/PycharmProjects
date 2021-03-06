# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'QMyMainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_QMyMainWindow(object):
    def setupUi(self, QMyMainWindow):
        QMyMainWindow.setObjectName("QMyMainWindow")
        QMyMainWindow.resize(2100, 800)
        font = QtGui.QFont()
        font.setPointSize(14)
        QMyMainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(QMyMainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.plainTextEdit.setGeometry(QtCore.QRect(60, 10, 491, 251))
        self.plainTextEdit.setMinimumSize(QtCore.QSize(2, 0))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(14)
        self.plainTextEdit.setFont(font)
        self.plainTextEdit.setObjectName("plainTextEdit")
        QMyMainWindow.setCentralWidget(self.centralwidget)
        self.menuBar = QtWidgets.QMenuBar(QMyMainWindow)
        self.menuBar.setGeometry(QtCore.QRect(0, 0, 800, 32))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.menuBar.setFont(font)
        self.menuBar.setObjectName("menuBar")
        self.menu_F = QtWidgets.QMenu(self.menuBar)
        self.menu_F.setMinimumSize(QtCore.QSize(16, 8))
        self.menu_F.setObjectName("menu_F")
        self.menu_E = QtWidgets.QMenu(self.menuBar)
        self.menu_E.setMinimumSize(QtCore.QSize(16, 8))
        self.menu_E.setObjectName("menu_E")
        self.menu_M = QtWidgets.QMenu(self.menuBar)
        self.menu_M.setMinimumSize(QtCore.QSize(16, 8))
        self.menu_M.setObjectName("menu_M")
        self.menu = QtWidgets.QMenu(self.menu_M)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.menu.setFont(font)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menu_M)
        self.menu_2.setObjectName("menu_2")
        QMyMainWindow.setMenuBar(self.menuBar)
        self.filetoolBar = QtWidgets.QToolBar(QMyMainWindow)
        self.filetoolBar.setMinimumSize(QtCore.QSize(5, 8))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.filetoolBar.setFont(font)
        self.filetoolBar.setOrientation(QtCore.Qt.Horizontal)
        self.filetoolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.filetoolBar.setObjectName("filetoolBar")
        QMyMainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.filetoolBar)
        self.actiontoolBar = QtWidgets.QToolBar(QMyMainWindow)
        self.actiontoolBar.setEnabled(False)
        self.actiontoolBar.setMinimumSize(QtCore.QSize(5, 8))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.actiontoolBar.setFont(font)
        self.actiontoolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.actiontoolBar.setObjectName("actiontoolBar")
        QMyMainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.actiontoolBar)
        self.texttoolBar = QtWidgets.QToolBar(QMyMainWindow)
        self.texttoolBar.setMinimumSize(QtCore.QSize(5, 8))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.texttoolBar.setFont(font)
        self.texttoolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.texttoolBar.setObjectName("texttoolBar")
        QMyMainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.texttoolBar)
        self.LanguagetoolBar = QtWidgets.QToolBar(QMyMainWindow)
        self.LanguagetoolBar.setEnabled(True)
        self.LanguagetoolBar.setMinimumSize(QtCore.QSize(5, 8))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.LanguagetoolBar.setFont(font)
        self.LanguagetoolBar.setAutoFillBackground(False)
        self.LanguagetoolBar.setToolButtonStyle(QtCore.Qt.ToolButtonTextUnderIcon)
        self.LanguagetoolBar.setObjectName("LanguagetoolBar")
        QMyMainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.LanguagetoolBar)
        self.statusBar = QtWidgets.QStatusBar(QMyMainWindow)
        self.statusBar.setObjectName("statusBar")
        QMyMainWindow.setStatusBar(self.statusBar)
        self.action_fileopen = QtWidgets.QAction(QMyMainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/images/images/images/001.GIF"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_fileopen.setIcon(icon)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.action_fileopen.setFont(font)
        self.action_fileopen.setObjectName("action_fileopen")
        self.action_fileclose = QtWidgets.QAction(QMyMainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/images/images/images/132.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_fileclose.setIcon(icon1)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.action_fileclose.setFont(font)
        self.action_fileclose.setObjectName("action_fileclose")
        self.action_textcut = QtWidgets.QAction(QMyMainWindow)
        self.action_textcut.setCheckable(False)
        self.action_textcut.setEnabled(False)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/images/images/images/200.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_textcut.setIcon(icon2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.action_textcut.setFont(font)
        self.action_textcut.setObjectName("action_textcut")
        self.action_textpaste = QtWidgets.QAction(QMyMainWindow)
        self.action_textpaste.setEnabled(False)
        icon3 = QtGui.QIcon()
        icon3.addPixmap(QtGui.QPixmap(":/images/images/images/204.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_textpaste.setIcon(icon3)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.action_textpaste.setFont(font)
        self.action_textpaste.setObjectName("action_textpaste")
        self.action_textcopy = QtWidgets.QAction(QMyMainWindow)
        self.action_textcopy.setCheckable(False)
        self.action_textcopy.setEnabled(False)
        icon4 = QtGui.QIcon()
        icon4.addPixmap(QtGui.QPixmap(":/images/images/images/202.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_textcopy.setIcon(icon4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.action_textcopy.setFont(font)
        self.action_textcopy.setObjectName("action_textcopy")
        self.action_bold = QtWidgets.QAction(QMyMainWindow)
        self.action_bold.setCheckable(True)
        icon5 = QtGui.QIcon()
        icon5.addPixmap(QtGui.QPixmap(":/images/images/images/500.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_bold.setIcon(icon5)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.action_bold.setFont(font)
        self.action_bold.setObjectName("action_bold")
        self.action_textitalic = QtWidgets.QAction(QMyMainWindow)
        self.action_textitalic.setCheckable(True)
        icon6 = QtGui.QIcon()
        icon6.addPixmap(QtGui.QPixmap(":/images/images/images/502.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_textitalic.setIcon(icon6)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.action_textitalic.setFont(font)
        self.action_textitalic.setObjectName("action_textitalic")
        self.action_textunderline = QtWidgets.QAction(QMyMainWindow)
        self.action_textunderline.setCheckable(True)
        icon7 = QtGui.QIcon()
        icon7.addPixmap(QtGui.QPixmap(":/images/images/images/504.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_textunderline.setIcon(icon7)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.action_textunderline.setFont(font)
        self.action_textunderline.setObjectName("action_textunderline")
        self.action_textshow = QtWidgets.QAction(QMyMainWindow)
        self.action_textshow.setCheckable(True)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.action_textshow.setFont(font)
        self.action_textshow.setObjectName("action_textshow")
        self.action_textclear = QtWidgets.QAction(QMyMainWindow)
        icon8 = QtGui.QIcon()
        icon8.addPixmap(QtGui.QPixmap(":/images/images/images/212.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_textclear.setIcon(icon8)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.action_textclear.setFont(font)
        self.action_textclear.setObjectName("action_textclear")
        self.action_textundo = QtWidgets.QAction(QMyMainWindow)
        self.action_textundo.setEnabled(False)
        icon9 = QtGui.QIcon()
        icon9.addPixmap(QtGui.QPixmap(":/images/images/images/206.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_textundo.setIcon(icon9)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.action_textundo.setFont(font)
        self.action_textundo.setObjectName("action_textundo")
        self.action_textredo = QtWidgets.QAction(QMyMainWindow)
        self.action_textredo.setEnabled(False)
        icon10 = QtGui.QIcon()
        icon10.addPixmap(QtGui.QPixmap(":/images/images/images/208.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_textredo.setIcon(icon10)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.action_textredo.setFont(font)
        self.action_textredo.setObjectName("action_textredo")
        self.action_textselectall = QtWidgets.QAction(QMyMainWindow)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.action_textselectall.setFont(font)
        self.action_textselectall.setObjectName("action_textselectall")
        self.action_filenew = QtWidgets.QAction(QMyMainWindow)
        self.action_filenew.setCheckable(False)
        icon11 = QtGui.QIcon()
        icon11.addPixmap(QtGui.QPixmap(":/images/images/images/100.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_filenew.setIcon(icon11)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.action_filenew.setFont(font)
        self.action_filenew.setObjectName("action_filenew")
        self.action_filesave = QtWidgets.QAction(QMyMainWindow)
        icon12 = QtGui.QIcon()
        icon12.addPixmap(QtGui.QPixmap(":/images/images/images/104.bmp"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_filesave.setIcon(icon12)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.action_filesave.setFont(font)
        self.action_filesave.setObjectName("action_filesave")
        self.action_English = QtWidgets.QAction(QMyMainWindow)
        self.action_English.setCheckable(True)
        icon13 = QtGui.QIcon()
        icon13.addPixmap(QtGui.QPixmap(":/images/images/images/timg2.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_English.setIcon(icon13)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.action_English.setFont(font)
        self.action_English.setObjectName("action_English")
        self.action_Chinese = QtWidgets.QAction(QMyMainWindow)
        self.action_Chinese.setCheckable(True)
        icon14 = QtGui.QIcon()
        icon14.addPixmap(QtGui.QPixmap(":/images/images/images/CN.jpg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.action_Chinese.setIcon(icon14)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.action_Chinese.setFont(font)
        self.action_Chinese.setObjectName("action_Chinese")
        self.menu_F.addAction(self.action_filenew)
        self.menu_F.addAction(self.action_fileopen)
        self.menu_F.addAction(self.action_filesave)
        self.menu_F.addAction(self.action_fileclose)
        self.menu_E.addAction(self.action_textcut)
        self.menu_E.addAction(self.action_textcopy)
        self.menu_E.addAction(self.action_textpaste)
        self.menu_E.addAction(self.action_textundo)
        self.menu_E.addAction(self.action_textredo)
        self.menu_E.addAction(self.action_textclear)
        self.menu_E.addAction(self.action_textselectall)
        self.menu.addAction(self.action_English)
        self.menu.addAction(self.action_Chinese)
        self.menu_M.addAction(self.action_bold)
        self.menu_M.addAction(self.action_textitalic)
        self.menu_M.addAction(self.action_textunderline)
        self.menu_M.addAction(self.action_textshow)
        self.menu_M.addAction(self.menu.menuAction())
        self.menu_M.addAction(self.menu_2.menuAction())
        self.menuBar.addAction(self.menu_F.menuAction())
        self.menuBar.addAction(self.menu_E.menuAction())
        self.menuBar.addAction(self.menu_M.menuAction())
        self.filetoolBar.addAction(self.action_fileopen)
        self.filetoolBar.addAction(self.action_filenew)
        self.filetoolBar.addAction(self.action_filesave)
        self.actiontoolBar.addAction(self.action_textcut)
        self.actiontoolBar.addAction(self.action_textcopy)
        self.actiontoolBar.addAction(self.action_textpaste)
        self.actiontoolBar.addAction(self.action_textundo)
        self.actiontoolBar.addAction(self.action_textredo)
        self.texttoolBar.addAction(self.action_bold)
        self.texttoolBar.addAction(self.action_textitalic)
        self.texttoolBar.addAction(self.action_textunderline)
        self.LanguagetoolBar.addAction(self.action_English)
        self.LanguagetoolBar.addAction(self.action_Chinese)

        self.retranslateUi(QMyMainWindow)
        self.action_textundo.triggered.connect(self.plainTextEdit.undo)
        self.action_textredo.triggered.connect(self.plainTextEdit.redo)
        self.action_textcut.triggered.connect(self.plainTextEdit.cut)
        self.action_textcopy.triggered.connect(self.plainTextEdit.copy)
        self.action_textpaste.triggered.connect(self.plainTextEdit.paste)
        self.action_textclear.triggered.connect(self.plainTextEdit.clear)
        self.action_textselectall.triggered.connect(self.plainTextEdit.selectAll)
        self.plainTextEdit.undoAvailable['bool'].connect(self.action_textundo.setEnabled)
        self.plainTextEdit.redoAvailable['bool'].connect(self.action_textredo.setEnabled)
        self.action_fileclose.triggered.connect(QMyMainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(QMyMainWindow)

    def retranslateUi(self, QMyMainWindow):
        _translate = QtCore.QCoreApplication.translate
        QMyMainWindow.setWindowTitle(_translate("QMyMainWindow", "QMyMainWindow"))
        self.menu_F.setTitle(_translate("QMyMainWindow", "??????(F)"))
        self.menu_E.setTitle(_translate("QMyMainWindow", "??????(E)"))
        self.menu_M.setTitle(_translate("QMyMainWindow", "??????(M)"))
        self.menu.setTitle(_translate("QMyMainWindow", "????????????"))
        self.menu_2.setTitle(_translate("QMyMainWindow", "???????????????"))
        self.filetoolBar.setWindowTitle(_translate("QMyMainWindow", "File"))
        self.actiontoolBar.setWindowTitle(_translate("QMyMainWindow", "Action"))
        self.texttoolBar.setWindowTitle(_translate("QMyMainWindow", "Edit"))
        self.LanguagetoolBar.setWindowTitle(_translate("QMyMainWindow", "Language"))
        self.action_fileopen.setText(_translate("QMyMainWindow", "??????..."))
        self.action_fileopen.setToolTip(_translate("QMyMainWindow", "????????????"))
        self.action_fileopen.setShortcut(_translate("QMyMainWindow", "Ctrl+O"))
        self.action_fileclose.setText(_translate("QMyMainWindow", "??????"))
        self.action_fileclose.setToolTip(_translate("QMyMainWindow", "????????????"))
        self.action_textcut.setText(_translate("QMyMainWindow", "??????"))
        self.action_textcut.setToolTip(_translate("QMyMainWindow", "??????"))
        self.action_textcut.setShortcut(_translate("QMyMainWindow", "Ctrl+X"))
        self.action_textpaste.setText(_translate("QMyMainWindow", "??????"))
        self.action_textpaste.setToolTip(_translate("QMyMainWindow", "??????"))
        self.action_textpaste.setShortcut(_translate("QMyMainWindow", "Ctrl+V"))
        self.action_textcopy.setText(_translate("QMyMainWindow", "??????"))
        self.action_textcopy.setToolTip(_translate("QMyMainWindow", "??????"))
        self.action_textcopy.setShortcut(_translate("QMyMainWindow", "Ctrl+C"))
        self.action_bold.setText(_translate("QMyMainWindow", "??????"))
        self.action_bold.setToolTip(_translate("QMyMainWindow", "??????"))
        self.action_bold.setShortcut(_translate("QMyMainWindow", "Ctrl+B"))
        self.action_textitalic.setText(_translate("QMyMainWindow", "??????"))
        self.action_textitalic.setToolTip(_translate("QMyMainWindow", "??????"))
        self.action_textitalic.setShortcut(_translate("QMyMainWindow", "Ctrl+I"))
        self.action_textunderline.setText(_translate("QMyMainWindow", "?????????"))
        self.action_textunderline.setToolTip(_translate("QMyMainWindow", "?????????"))
        self.action_textunderline.setShortcut(_translate("QMyMainWindow", "Ctrl+U"))
        self.action_textshow.setText(_translate("QMyMainWindow", "??????"))
        self.action_textshow.setToolTip(_translate("QMyMainWindow", "??????????????????"))
        self.action_textclear.setText(_translate("QMyMainWindow", "??????"))
        self.action_textclear.setToolTip(_translate("QMyMainWindow", "????????????"))
        self.action_textundo.setText(_translate("QMyMainWindow", "??????"))
        self.action_textundo.setToolTip(_translate("QMyMainWindow", "????????????"))
        self.action_textundo.setShortcut(_translate("QMyMainWindow", "Ctrl+Z"))
        self.action_textredo.setText(_translate("QMyMainWindow", "??????"))
        self.action_textredo.setToolTip(_translate("QMyMainWindow", "??????"))
        self.action_textredo.setShortcut(_translate("QMyMainWindow", "Ctrl+Y"))
        self.action_textselectall.setText(_translate("QMyMainWindow", "??????"))
        self.action_textselectall.setToolTip(_translate("QMyMainWindow", "??????????????????"))
        self.action_textselectall.setShortcut(_translate("QMyMainWindow", "Ctrl+A"))
        self.action_filenew.setText(_translate("QMyMainWindow", "??????"))
        self.action_filenew.setToolTip(_translate("QMyMainWindow", "??????"))
        self.action_filenew.setShortcut(_translate("QMyMainWindow", "Ctrl+N"))
        self.action_filesave.setText(_translate("QMyMainWindow", "??????"))
        self.action_filesave.setToolTip(_translate("QMyMainWindow", "????????????"))
        self.action_filesave.setShortcut(_translate("QMyMainWindow", "Ctrl+S"))
        self.action_English.setText(_translate("QMyMainWindow", "??????"))
        self.action_English.setToolTip(_translate("QMyMainWindow", "????????????"))
        self.action_Chinese.setText(_translate("QMyMainWindow", "??????"))
        self.action_Chinese.setToolTip(_translate("QMyMainWindow", "????????????"))
from GUI.Demo.Demo6 import QMyMainWindow_rc
