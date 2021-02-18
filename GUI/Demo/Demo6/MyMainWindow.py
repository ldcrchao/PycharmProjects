#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : MyMainWindow.py
@Author : chenbei
@Date : 2020/10/13 10:35
'''
'''
逻辑关系说明：
1、菜单栏
菜单栏是由工具栏设置好进行拖放的,所以底层菜单栏和工具栏就有关联,不需要任何槽函数或者代码进行关联
2、工具栏
①工具栏可以自定义名称、图标、提示文字以及快捷键
②可以使用工具栏的常用信号如trigger(bool)信号关联其他组件如文本组件
常用的槽函数有setEnabled(bool),可以根据文本的变化设置相应工具栏的使能
③文本组件的槽函数可以在内部直接关联好,也可以使用代码人为关联,人为关联时需要使用对应的C++槽函数
常用的文本槽函数有close、cut、copy、paste、redo、undo、clear、selectAll等
④文本组件常用信号有undoAvailable、redoAvailable、copyAvailable、customContextMenuRequested(快捷菜单)等
⑤可视化工具栏中不能直接定义其他类型(非QAction)的组件,如spinbox、combobox等,需要手动创建
⑥手动创建的组件内部没有槽函数,那就需要外部槽函数关联,以及组件与槽函数的关联语句
如本程序的self._spinfontsize.valueChanged[int].connect(self.do_fontsize_changed)
⑦对于spinbox常见信号为值改变信号valueChanged(),combobox则为下拉组件的索引改变信号currentIndexChanged()
具体可见Demo5、Demo4进行了解
3、状态栏
①新建、保存、打开、英语工具栏没有具体功能,这里只反映在状态栏的文字上
②状态栏引入了进度条,该进度条受spinbox的槽函数控制
'''
import sys
from PyQt5.QtWidgets import  (QApplication,QMainWindow,QActionGroup,QLabel,QProgressBar
                              ,QSpinBox ,QFontComboBox,QToolBar)
from PyQt5.QtCore import  Qt,pyqtSlot
from PyQt5.QtGui import  QTextCharFormat ,QFont,QIcon
from GUI.Demo.Demo6 import QMyMainWindow
class MyMainWindow(QMainWindow):
    def __init__(self,parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.ui = QMyMainWindow.Ui_QMyMainWindow()
        self.ui.setupUi(self)
        # 创建动态组件,添加到工具栏和状态栏，名称分别为filetoolBar,actiontoolBar,
        # texttoolBar,LanguagetoolBar,statusBar
        self._buildUI()
        self._spinfontsize.valueChanged[int].connect(self.do_fontsize_changed) # 自定义的动态组件必须手动连接槽函数
        self._combofontname.currentIndexChanged[str].connect(self.do_fontstyle_changed)
        self.setCentralWidget(self.ui.plainTextEdit) # 设置居中显示
        self.show()


    '''动态的添加字体类型和字体大小组件,字体类型关联到状态栏文本,字体大小关联到进度条'''
    def _buildUI(self):
        '''可视化设计中状态栏不能添加组件，工具栏也不能添加除Action类以外的组件
        只能通过代码对窗体添加动态组件
        另外界面语言的两个互斥Action必须用到QActionGroup分组才能实现互斥选择'''
        '''状态栏'''
        self._LabFile = QLabel(self) # 单继承法，继承标签类,可以用于显示来自文件名,字体名称,字体大小等
        self._LabFile.setMinimumWidth(150) # 标签最小宽度
        self._LabFile.setText("字体大小：")  # 设置初始状态标签文本
        self.ui.statusBar.addWidget(self._LabFile) # 添加一个静态文本组件到状态栏，表示来自哪个文件

        self._progressbar = QProgressBar(self) # 继承进度条类
        self._progressbar.setMaximumWidth(600) # 宽度要设置最大宽度，不然可能进度条与界面本身一样长
        # 不设置高度,自适应状态栏的高度
        self._progressbar.setMaximum(100)
        self._progressbar.setMinimum(5)
        sz = self.ui.plainTextEdit.font().pointSize() # 字体大小
        self._progressbar.setValue(sz) # 当前文本大小显示在进度条中
        self._progressbar.setFormat("%v")  # 设置显示当前值为步长模式, %p%为百分比模式
        self.ui.statusBar.addWidget(self._progressbar) # 设置在状态栏，表示字体大小的状态

        '''QActionGroup，窗体已有英语、中文组件，只需要设置互斥即可'''
        actiongroup = QActionGroup(self) # 互斥性分组类
        actiongroup.addAction(self.ui.action_English) # 添加动作
        actiongroup.addAction(self.ui.action_Chinese)
        actiongroup.setExclusive(True) # True表示互斥性分组
        self.ui.action_Chinese.setChecked(True) # 初始时刻默认中文突出显示

        '''texttoolBar类有粗体、斜体和下划线，没有字体大小和字体类型，这里使用滚动条调节字体大小，下拉条改变字体'''
        self._spinfontsize = QSpinBox(self) # 滚动条类
        self._spinfontsize.setMinimum(5) # 滚动条的上下限
        self._spinfontsize.setMaximum(100)
        sz1 = self.ui.plainTextEdit.font().pointSize()
        self._spinfontsize.setValue(sz1) # 设置滚动条初始状态为窗体设计时文本字体大小
        self._spinfontsize.setMinimumWidth(50) # 滚动条本身组件的宽度
        self._spinfontsize.setMinimumHeight(80)# 高度
        self.ui.texttoolBar.addWidget(self._spinfontsize) # 添加到工具栏

        self._combofontname = QFontComboBox(self) # 下拉条类，直接继承了字体类下拉条不需要自己设置
        self._combofontname.setMinimumWidth(250) # 本身组件宽度
        self.ui.texttoolBar.addWidget(self._combofontname) # 添加到工具栏

        # 根据MyMainWindow1.py文件的验证,可以得出结论：可以手动添加工具栏也可以直接窗体设计之初添加好
        self.ui.actiontoolBar.addAction(self.ui.action_textclear)
        self.ui.actiontoolBar.setEnabled(True) # 这里设置的是动作工具栏的使能
        #self.ui.action_textcopy.setEnabled(True)
        # 但是文本组件内容改变信号限制了复制、粘贴、剪贴的使能
        '''filetoolBar窗体已经设计好新建、保存、打开文件，但是没有关闭文件'''
        self.ui.filetoolBar.addSeparator() # 添加分割条
        self.ui.filetoolBar.addAction(self.ui.action_fileclose)  # 代码添加关闭按钮,窗体设计之初也可以直接设计好

        '''
        self._clear = QToolBar(self)
        self._clear.setMinimumWidth(50)
        self._clear.setMinimumHeight(80)
        bmp = QIcon(":/images/images/images/116.bmp")
        self._clear.setWindowIcon(bmp)
        self.ui.filetoolBar.addWidget(self._clear) # 手动添加清空工具栏
        '''

    '''combobox、spinbox动态工具栏没有内置槽函数只能使用外部槽函数,字体控制文本、字号控制状态栏的滑动条和文本'''
    @pyqtSlot(int)
    def do_fontsize_changed(self,fontsize): # 关联到字体大小滑动条
        '''滑动条的值fontsize改变信号连接到槽函数,槽函数负责设置改变文本的字体'''
        fmt = self.ui.plainTextEdit.currentCharFormat()
        fmt.setFontPointSize(fontsize)  # 滑动条的值去改变文本大小
        self.ui.plainTextEdit.mergeCurrentCharFormat(fmt)
        self._progressbar.setValue(fontsize) # 不仅关联文本字体,也关联进度条

    @pyqtSlot(str)
    def do_fontstyle_changed(self,fontname):# 关联到字体类型下拉条
        fmt = self.ui.plainTextEdit.currentCharFormat()
        fmt.setFontFamily(fontname)
        self.ui.plainTextEdit.mergeCurrentCharFormat(fmt)
        self._LabFile.setText("字体名称：%s"%fontname) # 也关联到状态标签

    '''内部C++槽函数,点击粗体、下划线、斜体工具栏控制文本'''
    @pyqtSlot(bool)
    def on_action_bold_triggered(self,checked):
        fmt = self.ui.plainTextEdit.currentCharFormat()
        if checked:
            fmt.setFontWeight(QFont.Bold)
        else:
            fmt.setFontWeight(QFont.Normal)
        self.ui.plainTextEdit.mergeCurrentCharFormat(fmt)
    @pyqtSlot(bool)
    def on_action_textitalic_triggered(self,checked):
        fmt = self.ui.plainTextEdit.currentCharFormat()
        fmt.setFontItalic(checked)
        self.ui.plainTextEdit.mergeCurrentCharFormat(fmt)
    @pyqtSlot(bool)
    def on_action_textunderline_triggered(self,checked):
        fmt = self.ui.plainTextEdit.currentCharFormat()
        fmt.setFontUnderline(checked)
        fmt.setUnderlineColor(Qt.red)
        self.ui.plainTextEdit.mergeCurrentCharFormat(fmt)

    '''文本组件的使能信号'''
    # 根据MyMainWindow1.py文件的验证,可以得出结论：可以手动设置使能函数也可以直接窗体设计之初全部定义好
    # 且如果不设置外部函数也不设置内部槽函数那么就不能够检测到变化
    def on_plainTextEdit_copyAvailable(self,checked):
        # 剪贴、复制和粘贴在选中文字以后才能使用
        # 撤销和重做使用的内部槽函数进行设置
        self.ui.action_textcopy.setEnabled(checked)
        self.ui.action_textcut.setEnabled(checked)
        self.ui.action_textpaste.setEnabled(checked)
        #pass

    def on_plainTextEdit_selectionChanged(self):
        # 选择不同的文字时会将相应文字当前格式反馈给按钮,设置是否突出显示
        # 如果不设置,若某个字有下划线和加粗,另一个只有加粗没有下划线,选中时下划线却会突出显示
        fmt = self.ui.plainTextEdit.currentCharFormat()
        self.ui.action_bold.setChecked(fmt.font().bold())
        self.ui.action_textitalic.setChecked(fmt.fontItalic())
        self.ui.action_textunderline.setChecked(fmt.fontUnderline())

    '''新建、保存、打开，控制状态栏，无具体功能'''
    #@pyqtSlot(bool)
    def on_action_filenew_triggered(self):
        self._LabFile.setText(" 新建文件 ")
    #@pyqtSlot(bool)
    def on_action_filesave_triggered(self):
        self._LabFile.setText(" 文件已保存 ")
    #@pyqtSlot(bool)
    def on_action_fileopen_triggered(self):
        self._LabFile.setText(" 文件已打开 ")
    '''显示、控制是否显示图标的文字'''
    @pyqtSlot(bool)
    def on_action_textshow_triggered(self,checked):# 在菜单栏上但是不在工具栏上
        if checked :
           st = Qt.ToolButtonTextUnderIcon
        else:
            st = Qt.ToolButtonIconOnly
        self.ui.filetoolBar.setToolButtonStyle(st)
        self.ui.texttoolBar.setToolButtonStyle(st)
        self.ui.actiontoolBar.setToolButtonStyle(st)
        self.ui.LanguagetoolBar.setToolButtonStyle(st)


# 主程序
app = QApplication(sys.argv)
bmp = QIcon(":/images/images/images/app.ico")
form = MyMainWindow()
form.setWindowIcon(bmp)
sys.exit(app.exec_())
