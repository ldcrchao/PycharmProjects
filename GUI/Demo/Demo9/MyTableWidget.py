#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : MyTableWidget.py
@Author : chenbei
@Date : 2020/11/6 18:31
'''
import sys
from PyQt5.QtWidgets import  (QApplication,QMainWindow,QLabel,QTableWidgetItem,QAbstractItemView)
from enum import Enum
from PyQt5.QtCore import pyqtSlot , Qt , QDate
from PyQt5.QtGui import  QFont ,QBrush ,QIcon
from GUI.Demo.Demo9 import QMyTableWidget
class CellType(Enum):
    ctHeader = 999 # 单元格类型
    ctName = 1000
    ctSex = 1001
    ctBirth = 1002
    ctNation = 1003
    ctScore = 1004
    ctParty = 1005
class FieldColNum(Enum):
     # 列编号
    colName = 0
    colSex = 1
    colBirth = 2
    colNation = 3
    colScore = 4
    colParty = 5
class MyTableWidget(QMainWindow):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.ui = QMyTableWidget.Ui_QMyTableWidget()
        self.ui.setupUi(self)
        self.LabCellIndex = QLabel("当前单元格坐标为:",self)
        self.LabCellIndex.setMaximumWidth(300)
        self.LabCellType = QLabel("当前单元格类型为:",self)
        self.LabCellType.setMinimumWidth(300)
        self.LabStudID = QLabel("学生ID:",self)
        self.LabStudID.setMinimumWidth(300)
        self.ui.statusbar.addWidget(self.LabCellIndex)
        self.ui.statusbar.addWidget(self.LabCellType)
        self.ui.statusbar.addWidget(self.LabStudID)
        self.ui.TabWidget.setAlternatingRowColors(True)
        self._tableinitialized = False
        self.ui.ShowLineHeader.setCheckState(Qt.Checked)
        self.ui.ShowListHeader.setCheckState(Qt.Checked)
        self.show()

    #设置表头
    @pyqtSlot()
    def on_SetHeader_clicked(self):
        HeaderText = ["姓 名","性 别","出生日期","民 族","分 数","是否党员"]
        self.ui.TabWidget.setColumnCount(len(HeaderText)) # 行表头,即列数
        for i in range(len(HeaderText)):
            HeaderItem = QTableWidgetItem(HeaderText[i],CellType.ctHeader.value) # 第一个参数是表格内容
            self.SetItemFont(HeaderItem,index=i) # 表头是表头类型单元格
    #设置行数
    @pyqtSlot()
    def on_SetRowsNum_clicked(self):
        self.ui.TabWidget.setRowCount(self.ui.spinBox.value()) # 把spinbox当前值赋给TableWidget设置行数
        self.ui.TabWidget.setAlternatingRowColors(self.ui.IntervalRowBackgroundColor.isChecked()) # 新建一行时要受到交替行底色按钮状态的控制
    #初始化表格
    @pyqtSlot()
    def on_InitialForm_clicked(self):
        self.ui.TabWidget.clearContents()
        birth = QDate(1997,9,1)
        isParty = True
        nation = "汉族"
        score = 50
        rowcount = self.ui.TabWidget.rowCount() # 表格行数
        for i in range(rowcount) :
            strName = "学生%d" % i
            score = score + 8 * i
            if ((i%2)==0) :
                strSex = "男"
            else:
                strSex = "女"
            self._CreateItemRow(i,strName,strSex,birth,nation,isParty,score) # 逐行传递的,传递的是整个行,所以后边是对一行的不同列进行设置的
            birth = birth.addDays(20)
            isParty = not isParty
        self._tableinitialized = True
    #初始化表格调用的函数
    def _CreateItemRow(self,index,name,sex,birth,nation,isParty,score):
        # 创建节点项函数QTableWidgetItem(text,type) 项的文字和类型
        # 设置节点项函数setItem(rowindex,coltype,item) 分别指定项的行和列,要添加的项
        #学号
        StudID = 2020090 + index
        #姓名
        itemName = QTableWidgetItem(name,CellType.ctName.value) # 第一个参数文本内容,第二个表格类型
        itemName.setData(Qt.UserRole, StudID)
        self.SetItemFont(itemName)
        # 由于逐行传递的,一行的不同列不仅要规定表格类型还要设置列编号,因为添加项有不同位置可以选择,必须指定编号
        self.ui.TabWidget.setItem(index,FieldColNum.colName.value,itemName) # 第一行的指定colName编号的列添加该项
        #性别
        itemSex = QTableWidgetItem(CellType.ctSex.value)
        self.SetItemFont(itemSex,sex=sex) # 传递sex是为了判定用哪个图标,而不是设置文字sex,创建时就已经设置好
        self.ui.TabWidget.setItem(index,FieldColNum.colSex.value,itemSex)
        #出生日期
        strBirth = birth.toString("yyyy-MM-dd")
        itemBirth = QTableWidgetItem(strBirth,CellType.ctBirth.value)
        self.SetItemFont(itemBirth)
        self.ui.TabWidget.setItem(index,FieldColNum.colBirth.value,itemBirth)
        #民族
        itemNation = QTableWidgetItem(nation,CellType.ctNation.value)
        self.SetItemFont(itemNation,nation=nation)
        self.ui.TabWidget.setItem(index,FieldColNum.colNation.value,itemNation)
        #分数
        strscore = str(score) # 传递来的score是int
        itemScore = QTableWidgetItem(strscore,CellType.ctScore.value)
        #if score < 60 :
        #    itemScore.setForeground(QBrush(Qt.green))
        self.SetItemFont(itemScore,score=score)
        self.ui.TabWidget.setItem(index,FieldColNum.colScore.value,itemScore)
        #党员
        itemParty = QTableWidgetItem("党员",CellType.ctParty.value)
        self.SetItemFont(itemParty,isParty=isParty)
        self.ui.TabWidget.setItem(index,FieldColNum.colParty.value,itemParty)
    #获取当前单元格的数据 内部C++槽函数信号
    @pyqtSlot(int,int,int,int)
    def on_TabWidget_currentCellChanged(self,currentrow,currentcol,previousrow,previouscol):
        if (self._tableinitialized == False ) :
            return  # 表格还没初始化时返回,放置空表格点击时执行程序发生错误
        item = self.ui.TabWidget.item(currentrow,currentcol) #找到当前项
        if item == None :
            return  # 为了防止点击到空项目也需要设置此语句保护不出错
        #if self.ui.RowSelection.isChecked() :
        #    return
        # 将单元格类型 显示到状态栏
        self.LabCellIndex.setText("当前单元格位置: %d 行  %d 列" %(currentrow+1,currentcol+1))
        itemCellType = item.type()
        self.LabCellType.setText("当前单元格类型: %d" % itemCellType )
        # 特别的当前单元格为名字类型时还显示学号到状态栏
        Item = self.ui.TabWidget.item(currentrow,FieldColNum.colName.value)
        self.LabStudID.setText("该学生的学号为: %d" % Item.data(Qt.UserRole))
    #插入行
    @pyqtSlot()
    def on_InsertRow_clicked(self):
        currentrow = self.ui.TabWidget.currentRow()
        self.ui.TabWidget.insertRow(currentrow)
        birth = QDate.fromString("1997-9-1","yyyy-M-d")
        self._CreateItemRow(currentrow,"李宁荣","男",birth,"苗族",True,59)
        self.ui.TabWidget.setAlternatingRowColors(self.ui.IntervalRowBackgroundColor.isChecked())#插入行需要受到间行底色限制
    #添加行
    @pyqtSlot()
    def on_AddRow_clicked(self):
        rowsNum = self.ui.TabWidget.rowCount()
        self.ui.TabWidget.insertRow(rowsNum) # 在最后的位置插入
        birth = QDate.fromString("2000-1-1","yyyy-M-d")
        self._CreateItemRow(rowsNum,"古力娜扎","女",birth,"维吾尔族",False,100)
        self.ui.TabWidget.setAlternatingRowColors(self.ui.IntervalRowBackgroundColor.isChecked())#添加行需要受到间行底色限制
    #删除当前行
    @pyqtSlot()
    def on_DeleteCurrentRow_clicked(self):
        self.ui.TabWidget.removeRow(self.ui.TabWidget.currentRow())
    #清空表格内容
    @pyqtSlot()
    def on_ClearTabContent_clicked(self):
        self.ui.TabWidget.clearContents()
    #自动调整列宽
    # 常用函数 resizeColumnsToContents() 自动调整所有列宽度,适应其内容
    # resizeColumnsToContents(column) 调整指定列的列宽
    @pyqtSlot()
    def on_AutoAdjustColWidth_clicked(self):
        self.ui.TabWidget.resizeColumnsToContents()
    #自动调整行高
    @pyqtSlot()
    def on_AutoAdjustRowHeight_clicked(self):
        self.ui.TabWidget.resizeRowsToContents()
    #表格可编辑
    @pyqtSlot(bool)
    def on_FormEditedEnabled_clicked(self,checked):
        if checked :
            # 可编辑则需要双击或者勾选方式
            trigger = (QAbstractItemView.DoubleClicked | QAbstractItemView.SelectedClicked)
        else:
            # 否则不可以触发
            trigger = QAbstractItemView.NoEditTriggers
        self.ui.TabWidget.setEditTriggers(trigger)  # 设置表格的可编辑属性
    #是否显示行表头
    @pyqtSlot(bool)
    def on_ShowLineHeader_clicked(self,checked):
        self.ui.TabWidget.horizontalHeader().setVisible(checked)
    #是否显示列表头
    @pyqtSlot(bool)
    def on_ShowListHeader_clicked(self,checked):
        self.ui.TabWidget.verticalHeader().setVisible(checked)
    #间隔行底色 (除了按钮触发,设置行数、插入行、添加行时也被间隔行底色按钮的勾选状态限制)
    @pyqtSlot(bool)
    def on_IntervalRowBackgroundColor_clicked(self,checked):
        self.ui.TabWidget.setAlternatingRowColors(checked)
    #选择模式 行选择模式
    @pyqtSlot()
    def on_RowSelection_clicked(self):
        self.ui.TabWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
    #选择模式 单元格选择模式
    @pyqtSlot()
    def on_CellSelection_clicked(self):
        self.ui.TabWidget.setSelectionBehavior(QAbstractItemView.SelectItems)
    #遍历表格读取数据
    @pyqtSlot()
    def on_ReadTabContentToText_clicked(self):
        self.ui.plainTextEdit.clear()
        if not self._tableinitialized :
            return
        if self.ui.TabWidget.item(0,0) == None :
            return
        rowcount = self.ui.TabWidget.rowCount()
        colcount = self.ui.TabWidget.columnCount()
        for i in range(rowcount) :
            strtext = "第 %d 行 : " %(i+1)
            for j in range(colcount-1) :
                cellitem = self.ui.TabWidget.item(i,j)
                strtext  = strtext + cellitem.text() + "   "
            cellitem = self.ui.TabWidget.item(i,colcount-1) # 特别的最后一列
            if (cellitem.checkState() == Qt.Checked): # 因为最后一列是checkbox类型不能读取文字,只能手动添加党员或者群众
               strtext = strtext + "党员"
            else :
                strtext = strtext + "群众"
            self.ui.plainTextEdit.appendPlainText(strtext)
    #设置项格式
    def SetItemFont(self, Item,**kwargs): # index=None,sex=None,nation=None,isParty=None
        font = QFont()  # 所有类型都具备的格式
        font.setPointSize(13)
        font.setFamily("Time New Roman")
        Item.setTextAlignment(Qt.AlignVCenter | Qt.AlignHCenter)

        for key in kwargs :
            if key == "index" : # 规定按照index、sex、nation、isParty、score
                index = kwargs[key] # 判断输入的动态参数是哪个
            elif key == "sex":  # 再将该动态参数赋给相应的参数
                sex = kwargs[key] # 相应的参数分别用于执行不同的程序,避免每次输入的动态参数只有一个而导致出现问题
            elif key == "nation":
                nation =kwargs[key]
            elif key == "isParty":
                isParty = kwargs[key]
            elif key == "score":
                score = kwargs[key]
        if Item.type() == CellType.ctHeader.value:
            self.ui.TabWidget.setHorizontalHeaderItem(index, Item)  # 表头类型设置水平居中
            Item.setForeground(QBrush(Qt.red))  # 表头文字颜色 红色
            Item.setFont(font)
        elif Item.type() == CellType.ctName.value:
            font.setBold(True)  # 名字类型 加粗
            Item.setFont(font)
            #Item.setData(Qt.UserRole, StudyID)  # 名字类型 关联学号
        elif Item.type() == CellType.ctSex.value:
            if sex =="男":
                icon = QIcon(":/images/images/boy.ico")
            else:
                icon = QIcon(":/images/images/girl.ico")
            Item.setIcon(icon)  # 性别类型 根据性别设置图标
            Item.setFont(font)
        elif Item.type() == CellType.ctBirth.value :
            Item.setFont(font) # 生日类型 无特殊格式
        elif Item.type() == CellType.ctNation.value :
            if nation != "汉族":
                Item.setForeground(QBrush(Qt.blue)) # 民族类型 不是汉族则显示为蓝色
            Item.setFont(font)
        elif Item.type() == CellType.ctScore.value :
            #print(score)
            if score < 60 : # 传递过来的是整型,但是进入此函数是score=None型,所以会出错
            #if score == 60 :
                Item.setForeground(QBrush(Qt.green))
            Item.setFont(font)  # 分数类型 低于60显示绿色
        elif Item.type() == CellType.ctParty.value :
            if isParty == True :
                Item.setCheckState(Qt.Checked) # 党员类型 为真则复选框打勾 相当于此时项变为一个复选类型的项
            else:
                Item.setCheckState(Qt.Unchecked)
            Item.setFlags(Qt.ItemIsSelectable|Qt.ItemIsEnabled|Qt.ItemIsUserCheckable) # 缺失Qt.ItemIsEditable,即党员类型单元格不允许编辑,只能勾选或者不勾选
            Item.setBackground(QBrush(Qt.yellow)) # 党员 背景色黄色
            Item.setFont(font)

app = QApplication(sys.argv)
bmp = QIcon(":/images/images/app.ico")
form = MyTableWidget()
form.setWindowIcon(bmp)
sys.exit(app.exec_())
