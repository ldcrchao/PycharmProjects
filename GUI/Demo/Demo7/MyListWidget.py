#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : MyListWidget.py
@Author : chenbei
@Date : 2020/10/16 18:30
'''
import sys
from PyQt5.QtWidgets import  (QApplication,QMainWindow,QListWidgetItem,QMenu,QToolButton)
from PyQt5.QtGui import QIcon,QFont,QCursor
from PyQt5.QtCore import  pyqtSlot ,Qt
from GUI.Demo.Demo7 import QMyListWidget
class MyListWidget(QMainWindow):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.ui = QMyListWidget.Ui_QMyListWidget()
        self.ui.setupUi(self)

        self.setCentralWidget(self.ui.splitter) # 使splitter充满整个工作区
        self._setActionForButton() # 关联函数：建立动作(工具栏)与工具按钮的关联
        self._createSelectionMenu() # 关联函数：建立动作(工具栏的"项选择")与工具按钮"项选择"的关联

        # 设置listwidget的项flag属性,这里表示1个集合同时设置了4种属性or3种
        # 项可被选择、可被编辑、可被复选、可使能
        self._FlagEditable = (Qt.ItemIsSelectable | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsEditable)
        self._FlagNotEditable = (Qt.ItemIsSelectable | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled )

        #self.ui.btn_initial.setText("初始化列表")
        self.ui.checkBox.setChecked(True) # 设计的时候遗漏了部分图标的设置，这里代码手动添加
        self.ui.action_additem.setIcon(QIcon(":/images/images/316.bmp"))
        self.ui.action_selectall.setIcon(QIcon(":/images/images/406.bmp"))

        self.setContextMenuPolicy(Qt.CustomContextMenu) # 此语句必须添加,否则不显示右键快捷菜单
        self.customContextMenuRequested.connect(self.on_listWidget_customContextMenuRequested)
        self.show()

    def _setActionForButton(self):
        '''动作关联工具按钮以后，点击toolbox的工具按钮即相当于点击工具栏的动作,执行工具栏关联的槽函数
        setDefaultAction()是QToolButton类的方法'''
        '''窗体左侧的工具按钮与动作的关联'''
        self.ui.btn_initial.setDefaultAction(self.ui.action_initial) # 初始化列表
        self.ui.btn_clear.setDefaultAction(self.ui.action_clear)  # 清空列表
        self.ui.btn_additem.setDefaultAction(self.ui.action_additem) # 添加项
        self.ui.btn_deleteitem.setDefaultAction(self.ui.action_deleteitem) # 删除项
        self.ui.btn_insertitem.setDefaultAction(self.ui.action_insertitem) # 插入项
        '''窗体右侧的工具按钮与动作的关联'''
        self.ui.btn_selectall.setDefaultAction(self.ui.action_selectall)
        self.ui.btn_selectitem_2.setDefaultAction(self.ui.action_quitselectall) # 取消全选
        self.ui.btn_selectitem_3.setDefaultAction(self.ui.action_selectinverse) # 反选
        # 还有一个“项选择”的工具按钮，由于是下拉菜单模式，这里单独进行关联，见createSelectionMenu()函数

    '''创建动作与外部槽函数的关联,"退出"和"选择项"已经分别内部关联了窗口的"close()"和反选"trigger()"槽函数'''
    '''其余动作的槽函数用于控制列表'''
    '''动作关联好槽函数后，其实也就是toolbox和group的(5+3)个QToolButoon建立好关联槽函数
    因为toolbox/groupbox中的工具按钮在_setActionForButton()函数中已经关联了动作'''
    # 初始化列表动作控制listwidget
    @pyqtSlot()
    def on_action_initial_triggered(self):
        icon1 = QIcon(":/images/images/BLD.BMP")
        icon2 = QIcon(":/images/images/502.bmp")
        icon3 = QIcon(":/images/images/504.bmp")
        icon4 = QIcon(":/images/images/508.bmp")
        icon5 = QIcon(":/images/images/512.bmp")
        icon6 = QIcon(":/images/images/510.bmp")
        icon7 = QIcon(":/images/images/310.bmp")
        icon8 = QIcon(":/images/images/312.bmp")
        icon9 = QIcon(":/images/images/328.bmp")
        icon10 = QIcon(":/images/images/418.bmp")
        icon11 = QIcon(":/images/images/324.bmp")
        icon = [icon1,icon2 ,icon3,icon4,icon5 ,icon6,icon7,icon8,icon9,icon10,icon11]
        string = ["粗体","斜体","下划线","靠左对齐","靠右对齐","居中对齐","左移","右移","问号","放大","退出"]
        editable = self.ui.checkBox.isChecked() # 复选编辑框勾上以后可以进行
        if editable :
            Flag = self._FlagEditable
        else:
            Flag = self._FlagNotEditable
        alignment = Qt.AlignHCenter
        font = QFont()
        font.setFamily("TimeNewRoman")
        font.setPointSize(15)
        #Flag = self.on_checkBox_clicked(editable)
        self.ui.listWidget.clear()  # 清除列表
        for i in range(len(icon)):
            item = QListWidgetItem()
            item.setText(string[i])
            item.setIcon(icon[i])
            item.setCheckState(Qt.Checked)
            item.setFlags(Flag)
            item.setFont(font)
            item.setTextAlignment(alignment)
            self.ui.listWidget.addItem(item)

    # 复选框控制列表项的可编辑属性
    @pyqtSlot(bool)
    def on_checkBox_clicked(self,checked):
        if checked :
            Flag = self._FlagEditable
        else:
            Flag = self._FlagNotEditable
        item = self.ui.listWidget.currentItem()
        #item.setCheckState(checked)
        if item :
           item.setFlags(Flag)
        #return Flag

    # 添加项动作控制列表项
    @pyqtSlot()
    def on_action_additem_triggered(self):
        icon = QIcon(":/images/images/718.bmp")
        editable = self.ui.checkBox.isChecked()
        if editable :
            Flag = self._FlagEditable
        else:
            Flag = self._FlagNotEditable
        font = QFont()
        font.setPointSize(15)
        font.setFamily("TimeNewRoman")
        #Flag = self.on_checkBox_clicked(editable)
        item = QListWidgetItem()
        item.setText("红灯") # 添加的项目
        item.setTextAlignment(Qt.AlignHCenter)
        item.setIcon(icon)
        item.setFont(font)
        item.setCheckState(Qt.Checked)
        item.setFlags(Flag)
        self.ui.listWidget.addItem(item)
    # 插入项动作控制列表项
    @pyqtSlot()
    def on_action_insertitem_triggered(self):
        icon = QIcon(":/images/images/724.bmp")
        editable = self.ui.checkBox.isChecked()
        if editable :
            Flag = self._FlagEditable
        else:
            Flag = self._FlagNotEditable
        font = QFont()
        font.setPointSize(15)
        font.setFamily("TimeNewRoman")
        #Flag = self.on_checkBox_clicked(editable)
        item = QListWidgetItem()
        item.setText("绿灯") # 添加的项目
        item.setTextAlignment(Qt.AlignHCenter)
        item.setIcon(icon)
        item.setFont(font)
        item.setCheckState(Qt.Checked)# 初始化为选中状态
        item.setFlags(Flag)
        currentrow = self.ui.listWidget.currentRow()
        # insertItem(self,row,itemtext) or insertItem(self,row,item)
        # 在当前行之后插入项，前者只能插入文本不能设置其他属性,后者需要实例化项类
        self.ui.listWidget.insertItem(currentrow,item) # 后者

    # 删除项动作控制列表项
    @pyqtSlot()
    def on_action_deleteitem_triggered(self):
        row = self.ui.listWidget.currentRow()
        self.ui.listWidget.takeItem(row)

    # 清空动作控制列表
    @pyqtSlot()
    def on_action_clear_triggered(self):
        self.ui.listWidget.clear()

    # 全选动作控制列表项
    @pyqtSlot()
    def on_action_selectall_triggered(self):
        for i in range(self.ui.listWidget.count()):
            Item = self.ui.listWidget.item(i)
            Item.setCheckState(Qt.Checked)

    # 全不选动作控制列表项
    @pyqtSlot()
    def on_action_quitselectall_triggered(self):
        for i in range(self.ui.listWidget.count()):
            Item = self.ui.listWidget.item(i)
            Item.setCheckState(Qt.Unchecked)

    # 反选动作控制列表项
    @pyqtSlot()
    def on_action_selectinverse_triggered(self):
        for i in range(self.ui.listWidget.count()):
            Item = self.ui.listWidget.item(i)
            if (Item.checkState() != Qt.Checked) :
                Item.setCheckState(Qt.Checked)
            else:
                Item.setCheckState(Qt.Unchecked)

    '''列表当前项控制文本'''
    def on_listWidget_currentItemChanged(self,currentitem,previousitem):
        '''列表变化的内部信号,带有两个参数分别表示前一项和后一项'''
        strinfo = ""
        if (currentitem!=None):
            if (previousitem == None):# 也就是第1项,只显示第一项
                strinfo ="Current:" + currentitem.text()
            else:
                strinfo = "Previous:" + previousitem.text() +"\n"+ "Current:" + currentitem.text()
        self.ui.lineEdit.setText(strinfo)

    # 到现在为止窗体的工具按钮除"项选择"之外都已经关联好了外部槽函数
    # 复选框、文本组件也已经使用
    # 由于下拉菜单的缘故现在对"项选择"需要单独进行设置
    def _createSelectionMenu(self):
        # 首先需要构建下拉菜单
        # 菜单中就关联好了动作，所以下方的工具按钮添加了菜单就等于联系了动作
        menuselection = QMenu() # 实例化的类可以被工具栏的"项选择"和QListWidget的"项选择"引用
        menuselection.addAction(self.ui.action_selectall) # 下拉菜单联系了全选、取消全选、反选
        menuselection.addAction(self.ui.action_selectinverse)
        menuselection.addAction(self.ui.action_quitselectall)

        ''' QListWidget的"项选择"工具按钮菜单设置 '''
        # 窗体设置中已经设置好了该QToolButton为下拉菜单模式，但是以防万一还是添加该语句初始化
        # MenuButtonPopup模式 直接单击按钮会执行按钮关联的Action，而不会弹出下拉菜单，必须点击小箭头才能弹出菜单
        self.ui.btn_selectitem.setPopupMode(QToolButton.MenuButtonPopup)
        # 此种模式点击按钮时直接出现下拉菜单，不会触发Action
        #self.ui.btn_selectitem.setPopupMode(QToolButton.InstantPopup)
        self.ui.btn_selectitem.setToolButtonStyle(Qt.ToolButtonTextBesideIcon) # 图标靠左
        self.ui.btn_selectitem.setMenu(menuselection) # 引用实例化的菜单类

        ''' 工具栏的"项选择"动作菜单设置 '''
        # 书中是因为窗体设计时没有将"项选择"工具按钮添加到工具栏，所以需要实例化工具按钮，构建菜单再添加到工具栏
        # 工具栏的动作本身是不能添加菜单的,所以需要实例化一个动作工具按钮添加至工具栏才能添加菜单
        # 动作action_selection为了能够添加菜单只能依附到实例化的工具按钮toolBtn,也就是说实际上工具栏的"项选择"动作不再是动作而变成了1个真正的按钮
        # 只是动作和按钮之间已经建立了联系
        ## 由于窗体之初就有了“项选择”动作可视组件，以下新添加可能会导致出现2个"项选择"
        toolBtn = QToolButton(self) # 需要继承窗体本身
        toolBtn.setPopupMode(QToolButton.InstantPopup)
        toolBtn.setDefaultAction(self.ui.action_selectitem) # 实例化的工具按钮与工具栏的"项选择"动作建立关联
        toolBtn.setMenu(menuselection)
        toolBtn.setText("项选择")
        font = QFont()
        font.setFamily("TimeNewRoman")
        font.setPointSize(14)
        toolBtn.setFont(font)
        toolBtn.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.ui.toolBar.addWidget(toolBtn)

        '''原文窗体设置之初没有把退出动作添加,这里手动添加'''
        #self.ui.toolBar.addSeparator()
        #self.ui.toolBar.addAction(self.ui.action_exit)

    def on_listWidget_customContextMenuRequested(self,pos):
        menu = QMenu(self)
        menu.addAction(self.ui.action_initial)
        menu.addAction(self.ui.action_selectall)
        menu.addAction(self.ui.action_quitselectall)
        menu.addAction(self.ui.action_selectinverse)
        menu.addAction(self.ui.action_clear)
        menu.addSeparator()
        menu.addAction(self.ui.action_insertitem)
        menu.addAction(self.ui.action_additem)
        menu.addAction(self.ui.action_deleteitem)
        menu.addSeparator()
        menu.addAction(self.ui.action_exit )
        menu.exec(QCursor.pos()) # 显示菜单 QCursor.pos()会获取鼠标光标当前位置

# 主程序
app = QApplication(sys.argv)
bmp = QIcon(":/images/images/app.ico")
form = MyListWidget()
form.setWindowIcon(bmp)
sys.exit(app.exec_())











