# %%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : girlfriend.py
@Author : chenbei
@Date : 2020/11/1 19:00
'''
import os
import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QTreeWidgetItem,
    QFileDialog,
    QDockWidget)
from enum import Enum  # 枚举类型
from PyQt5.QtCore import pyqtSlot, Qt, QDir, QFileInfo
from PyQt5.QtGui import QIcon, QPixmap, QFont
from GUI.Demo.Demo8 import QMyTreeWidget


class TreeWidgetItem(Enum):
    itTopItem = 10   # 顶层节点
    itGroupItem = 11  # 分组节点
    itImageItem = 12  # 图片文件节点


class TreeColNum(Enum):
    colItem = 0   # 分组 / 文件名列 , 表示第1列
    colItemType = 1  # 节点类型列


class MyTreeWidget(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = QMyTreeWidget.Ui_QMyTreeWidget()
        self.ui.setupUi(self)

        self.curPixmap = QPixmap()  # 图片 , 用于存储当前显示的原始图片,图片的放大、缩小等都基于该原始图片
        self.pixRatio = 1  # 显示比例
        self.itemFlags = (Qt.ItemIsSelectable | Qt.ItemIsUserCheckable |
                          Qt.ItemIsEnabled | Qt.ItemIsAutoTristate)
        self.setCentralWidget(self.ui.ScrollArea)
        self.ui.treeWidget.header().setDefaultSectionSize(600)  # 设置表头两列之间的距离
        self.ui.treeWidget.header().setMinimumSectionSize(600)

        # 停靠区的属性,包括停靠区可关闭、停靠区可移动、停靠区可浮动、在停靠区左侧显示垂直标题栏
        self.ui.dockWidget.setFeatures(QDockWidget.AllDockWidgetFeatures)
        self.ui.dockWidget.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)  # 允许停靠区域 , 左右浮动
        self.ui.ScrollArea.setWidgetResizable(True)  # 自动调整scrollarea内部组件的大小
        self.ui.ScrollArea.setAlignment(Qt.AlignVCenter)  # 区域、图片均水平垂直居中
        self.ui.ScrollArea.setAlignment(Qt.AlignHCenter)
        self.ui.Pictures.setAlignment(Qt.AlignVCenter)
        self.ui.Pictures.setAlignment(Qt.AlignHCenter)
        self.ui.dockWidget.setFloating(False)
        # self.ui.Pictures.setScaledContents(True)
        self._intTree()  # 初始化树目录
        # self._intpictures() # 初始化静态图片
        self._inirespictures()
        self.show()

    def setitemfont(self, item, colItemName, fullfilename=None):
        '''
        :param item: 项
        :param colItemName: 项名字
        :param fullfilename: 当为图片节点时还需要指定图片地址
        :return: 设定项格式
        '''
        # setIcon和setText函数都需要给定列名,指定是对哪列进行设置
        # 同一节点有colItem和colItemType两列
        # 列名可以使用数字 , 为了便于统一修改这里定义了枚举类型TreeWidgetItem和TreeColNum
        # setData(self,column,role,value) 为某列设置一个角色数据,这里Qt.UserRole是枚举类型Qt.ItemDataRole中预定义的值,设置了1个空字符串数据
        #item.setData(TreeColNum.colItem.value,Qt.UserRole ,"")
        icon1 = QIcon(":/images/icons/31.ico")
        icon2 = QIcon(":/images/icons/Documents.ico")
        font = QFont()
        font.setFamily("Time New Roman")
        font.setPointSize(16)
        item.setText(TreeColNum.colItem.value, colItemName)
        if (item.type() == TreeWidgetItem.itTopItem.value):
            item.setText(TreeColNum.colItemType.value, "TopGroup")
            item.setIcon(TreeColNum.colItem.value, icon2)
        elif (item.type() == TreeWidgetItem.itGroupItem.value):
            item.setText(TreeColNum.colItemType.value, "Group")
            item.setIcon(TreeColNum.colItem.value, icon2)
        elif(item.type() == TreeWidgetItem.itImageItem.value):
            item.setText(TreeColNum.colItemType.value, "JPG")
            item.setIcon(TreeColNum.colItem.value, icon1)
            item.setData(TreeColNum.colItem.value, Qt.UserRole, fullfilename)
            # print(fullfilename)
        item.setTextAlignment(
            TreeColNum.colItemType.value,
            Qt.AlignVCenter | Qt.AlignHCenter)
        item.setFont(TreeColNum.colItemType.value, font)
        item.setFont(TreeColNum.colItem.value, font)
        item.setFlags(self.itemFlags)
        item.setCheckState(TreeColNum.colItem.value, Qt.Checked)  # 复选框

    def _intTree(self):
        self.ui.treeWidget.clear()
        paritem = self.ui.treeWidget.currentItem()
        if paritem is None:
            # 如果没有父节点或者没有选中父节点 不可以添加文件和删除节点
            self.ui.action_AddFile.setEnabled(False)
            # 由于初始化自动创建了父节点所以可以添加目录 , 设置使能为False没有意义
            self.ui.action_DeleteItem.setEnabled(False)
        TopItem = QTreeWidgetItem(TreeWidgetItem.itTopItem.value)
        self.setitemfont(TopItem, "我的朋友们")
        self.ui.treeWidget.addTopLevelItem(TopItem)
        friends = ["中国女朋友", "韩国女朋友", "日本女朋友", "欧美女朋友", "我的男朋友", "我的女朋友"]
        for i in range(len(friends)):
            GroupItem = QTreeWidgetItem(TreeWidgetItem.itGroupItem.value)
            self.setitemfont(GroupItem, friends[i])
            TopItem.addChild(GroupItem)

    def _inirespictures(self):
        add1 = ":/pictures/girls/"
        friends = ["中国女朋友", "韩国女朋友", "日本女朋友", "欧美女朋友", "我的男朋友", "我的女朋友"]
        add2 = []  # 每个元素存放类似 : :/pictures/girls/中国女朋友
        for i in range(len(friends)):
            temp = add1 + friends[i]
            add2.append(temp)
        chinesegirls = [
            '王晓晨.jpg',
            '柳岩.jpg',
            '宋祖儿.jpg',
            '佟丽娅.jpg',
            '刘诗诗.jpg',
            '古力娜扎.jpg',
            '吴宣仪.jpg',
            '周洁琼.jpg',
            '杨超越.jpg',
            '白鹿.jpg',
            '祝绪丹.jpg',
            '程潇.jpg',
            '范冰冰.jpg',
            '谭松韵.jpg',
            '赵丽颖.jpg',
            '赵露思.jpg',
            '陈钰琪.jpg']
        koreagirls = [
            '李知恩.jpg',
            '全智贤.jpg',
            '宋慧乔.jpg',
            '宋智孝.jpg',
            '朴信惠.jpg',
            '朴孝敏.jpg',
            '朴智妍.jpg',
            '林允儿.jpg',
            '赵宝儿.jpg',
            '金泰希.jpg',
            '金智秀.jpg']
        japangirls = [
            '苍井空.jpg',
            '坂井泉水.jpg',
            '新垣结衣.jpg',
            '桥本环奈.jpg',
            '石原里美.jpg',
            '西野翔.jpg']
        usa_eurogirls = ['泰勒.jpg']
        mygirls = ['古力娜扎.jpg']
        myboys = ['胡歌.jpg', '彭于晏.jpg', '吴彦祖.jpg']
        allgirls = [
            chinesegirls,
            koreagirls,
            japangirls,
            usa_eurogirls,
            myboys,
            mygirls]
        for j in range(len(friends)):  # 中国女朋友时
            Groupitem = self.ui.treeWidget.topLevelItem(
                0).child(j)  # 找到中国女朋友节点
            girls = allgirls[j]  # 节点的不同找到对应的文件夹下所有图片的后缀
            # print(girls)
            for k in range(len(girls)):  # 中国女朋友时就会循环13次
                fullname = add2[j] + '/' + girls[k]
                # print(fullname)
                myitem = QTreeWidgetItem(TreeWidgetItem.itImageItem.value)
                self.setitemfont(myitem, str(girls[k]), fullname)
                Groupitem.addChild(myitem)

    def _intpictures(self):
        #addddd = ":/pictures/girlfriends_pictures/中国女朋友/古力娜扎.jpg"
        #myitem = QTreeWidgetItem(TreeWidgetItem.itImageItem.value)
        #topitem = self.ui.treeWidget.topLevelItem(0)
        #self.setitemfont(myitem, "My Girl", fullfilename=addddd )
        # topitem.addChild(myitem)
        Address_temp = os.path.abspath(
            r"C:/Users/chenbei/PycharmProjects/python学习工程文件夹/GUI/icons/girlfriends")
        #Address_temp = ":/pictures/girlfriends_pictures"
        friends = ["中国女朋友", "韩国女朋友", "日本女朋友", "欧美女朋友", "我的男朋友", "我的女朋友"]
        Address = []
        for i in range(len(friends)):
            tempaddress = Address_temp + '/' + (friends[i])
            # print(tempaddress)
            Address.append(tempaddress)
        #print(Address )
        for i in range(len(Address)):
            fileadd = os.listdir(Address[i])
            # print(fileadd)
            fileaddress = []
            for j in range(len(fileadd)):
                fileaddress.append(os.path.join(Address[i], fileadd[j]))
            # print(fileaddress)
            Groupitem = self.ui.treeWidget.topLevelItem(
                0).child(i)   # 找到对应文件夹节点
            for k in range(len(fileaddress)):
                fullFilename = fileaddress[k]
                #print(fullFilename )
                fileinfo = QFileInfo(fullFilename)
                nodetext = fileinfo.fileName()
                girlitem = QTreeWidgetItem(TreeWidgetItem.itImageItem.value)
                # print(fullFilename)
                self.setitemfont(girlitem, nodetext, fullfilename=fullFilename)
                Groupitem.addChild(girlitem)

    '''以下依次为QAction动作的内部C++槽函数设置 : 添加目录、新建文件、删除节点、遍历节点'''
    '''由于节点类型不同,其添加目录、新建文件、删除节点的操作有相应的使能,这里利用treewidget的值改变信号currentItemChanged()进行控制'''
    @pyqtSlot()
    def on_action_AddFolder_triggered(self):
        dirStr = QFileDialog.getExistingDirectory()  # 返回选择的目录类似于下边注释所示
        # dirStr = C:/Users/chenbei/PycharmProjects/python学习工程文件夹/特征提取
        # print(dirStr)
        if dirStr == "":
            # print("取消新建目录")
            return
        paritem = self.ui.treeWidget.currentItem()
        if paritem is None:
            paritem = self.ui.treeWidget.topLevelItem(0)
            # print(paritem)
        dirobj = QDir(dirStr)
        # print(dirobj) # <PyQt5.QtCore.QDir object at 0x00000200D5C2C588>
        nodetext = dirobj.dirName()  # 最后一级目录的名称,例如注释当中的"特征提取"
        # print(f"{nodetext}")
        item = QTreeWidgetItem(TreeWidgetItem.itGroupItem.value)
        self.setitemfont(item, nodetext, dirobj)
        paritem.addChild(item)  # 添加子节点
        paritem.setExpanded(True)  # 展开节点

    @pyqtSlot()
    def on_action_AddFile_triggered(self):
        # 设置文件扩展名过滤,用双分号间隔。eg : "All Files (*);;PDF Files (*.pdf);;Text Files
        # (*.txt)
        filelist, flt = QFileDialog.getOpenFileNames(
            self, "选择一个或多个文件", "", "Images(*.jpg)")
        # print(flt) # Images(*.jpg)
        #print("选择的文件数量为" + str(len(filelist)) + "个")
        # 新建图片文件时的对话框会出现提示信息"选择一个或多个文件" ,"Images(*.jpg)"则限制了可以选择的文件类型
        if (len(filelist) < 1):
            # print("选择文件数量不能少于1个")
            return
        item = self.ui.treeWidget.currentItem()
        if item.type() is None:
            return
        if (item.type() == TreeWidgetItem.itImageItem.value):  # 如果是图片节点则找到其父类节点,否则父节点就是自己
            paritem = item.parent()
        else:
            paritem = item
        for i in range(len(filelist)):
            fullFilename = filelist[i]  # 完整路径名
            # print(filelist[i] ) #
            # C:/Users/chenbei/PycharmProjects/python学习工程文件夹/GUI/icons/165.JPG
            fileinfo = QFileInfo(fullFilename)  # 在python当中保存的路径
            # print(fileinfo) # <PyQt5.QtCore.QFileInfo object at
            # 0x000001BDBD014518>
            nodetext = fileinfo.fileName()  # 路径最后一级的文件名
            # print(nodetext) # 165.JPG
            item = QTreeWidgetItem(TreeWidgetItem.itImageItem.value)
            self.setitemfont(item, nodetext, fullFilename)
            paritem.addChild(item)
        paritem.setExpanded(True)

    # 使能控制
    def on_treeWidget_currentItemChanged(self, current, previous):
        # print(current) # 此语句就可以返回当前项属于何种枚举类型,顶层节点10,次级节点11,图片节点12
        if (current is None):
            # print("0")
            # self.ui.action_AddFolder.setEnabled(True)
            # self.ui.action_AddFile.setEnabled(False)  # 运行初始化的时候为了防止新建错误,设定使能为False
            # self.ui.action_DeleteItem(False)
            return
        nodeType = current.type()
        # print(nodeType)
        if (nodeType == TreeWidgetItem.itTopItem.value):
            self.ui.action_AddFolder.setEnabled(True)
            self.ui.action_AddFile.setEnabled(True)
            self.ui.action_DeleteItem.setEnabled(False)  # 顶层节点不能删除
            # print("1")
        elif (nodeType == TreeWidgetItem.itGroupItem.value):
            self.ui.action_AddFolder.setEnabled(True)
            self.ui.action_AddFile.setEnabled(True)
            self.ui.action_DeleteItem.setEnabled(True)  # 次级节点可以删除
            # print("2")
        elif (nodeType == TreeWidgetItem.itImageItem.value):
            self.ui.action_AddFolder.setEnabled(False)  # 终端节点不能添加目录
            self.ui.action_AddFile.setEnabled(False)
            self.ui.action_DeleteItem.setEnabled(True)
            # print("3")
            #filename = current.data(TreeColNum.colItem.value,Qt.UserRole)
            # print(filename)
            self._displayImage(current)

    @pyqtSlot()
    def on_action_DeleteItem_triggered(self):
        item = self.ui.treeWidget.currentItem()
        paritem = item.parent()  # 节点不能移除自己,必须使用其父节点
        if paritem is None:
            # print("顶层节点不能删除")
            return
        else:
            paritem.removeChild(item)

    @pyqtSlot()
    def on_action_save_triggered(self):
        TopGroupItem = self.ui.treeWidget.topLevelItem(0)
        self._iniLastTopGroupItem(TopGroupItem)
        GroupItemCount = TopGroupItem.childCount()  # 次级节点个数 = 6
        for i in range(GroupItemCount):
            GroupItem = TopGroupItem.child(i)
            self._iniLastGroupItem(TopGroupItem, GroupItem)
            ImageItemCount = GroupItem.childCount()  # 每个次级节点的终端节点个数 = 17、11、6、1、3、1
            # print(ImageItemCount)
            for j in range(ImageItemCount):
                ImageItem = GroupItem.child(j)
                self._iniLastImageItem(GroupItem, ImageItem)
                #print(ImageItem.data(TreeColNum.colItem.value,Qt.UserRole) )

    '''
    def _iniLastSave(self,item):
        if item.type() == TreeWidgetItem.itTopItem.value :
           self._iniLastTopGroupItem(item)
        elif item.type() == TreeWidgetItem.itGroupItem.value :
           self._iniLastGroupItem(item)
        elif item.type() == TreeWidgetItem.itImageItem.value:
           self._iniLastImageItem(item)
    '''

    def _iniLastTopGroupItem(self, topgroupitem):
        self.ui.treeWidget.addTopLevelItem(topgroupitem)
        print(topgroupitem.data(TreeColNum.colItem.value, Qt.UserRole))

    def _iniLastGroupItem(self, topgroupitem, groupitem):
        TopGroupItem = topgroupitem
        TopGroupItem.addChild(groupitem)

    def _iniLastImageItem(self, groupitem, imageitem):
        GroupItem = groupitem
        GroupItem.addChild(imageitem)

    @pyqtSlot()
    def on_action_ScanItems_triggered(self):
        count = self.ui.treeWidget.topLevelItemCount()  # 顶层节点个数
        # print(count)  # 1个
        for i in range(count):
            item = self.ui.treeWidget.topLevelItem(i)  # 索引某个顶层节点
            self._changeItemCaption(item)

    def _changeItemCaption(self, item):
        title = "*" + item.text(TreeColNum.colItem.value)  # 遍历一次就会加"*"一次
        item.setText(TreeColNum.colItem.value, title)
        # print(item.childCount())
        if (item.childCount() > 0):
            for i in range(item.childCount()):
                self._changeItemCaption(item.child(i))

    def _displayImage(self, item):
        filename = item.data(
            TreeColNum.colItem.value,
            Qt.UserRole)  # 通过setData设置的个人用户数据可通过data取出
        # print(type(filename))
        # print(filename) # 完整路径名称会在状态栏中显示
        self.ui.statusbar.showMessage(filename)
        self.ui.ScrollArea.setMinimumSize(850, 1000)  # 手动调节scrollarea的最小宽度与高度
        self.ui.Pictures.setMinimumSize(850, 1000)  # 最小宽度和高度
        self.curPixmap.load(filename)  # 原始图片
        self.ui.Pictures.setAlignment(Qt.AlignVCenter)
        self.ui.Pictures.setAlignment(Qt.AlignHCenter)
        self.on_action_adjustheight_triggered()
        self.on_action_adjustwidth_triggered()
        self.ui.action_adjustheight.setEnabled(True)
        self.ui.action_adjustwidth.setEnabled(True)
        self.ui.action_ZoomIn.setEnabled(True)
        self.ui.action_ZoomOut.setEnabled(True)
        self.ui.action_ZoomRealSize.setEnabled(True)

    '''以下设置放大、缩小、恢复、调整宽度、调整高度5个动作'''
    @pyqtSlot()
    def on_action_ZoomIn_triggered(self):
        self.pixRatio = self.pixRatio * 1.2
        W = self.pixRatio * self.curPixmap.width()
        H = self.pixRatio * self.curPixmap.height()
        # print("放大后的宽度为:"+str(W))
        # print("放大后的高度为:"+str(H))
        pix = self.curPixmap.scaled(W, H)
        self.ui.Pictures.setPixmap(pix)
        self.ui.Pictures.setAlignment(Qt.AlignVCenter)
        self.ui.Pictures.setAlignment(Qt.AlignHCenter)

    @pyqtSlot()
    def on_action_ZoomOut_triggered(self):
        self.pixRatio = self.pixRatio * 0.8
        W = self.pixRatio * self.curPixmap.width()
        H = self.pixRatio * self.curPixmap.height()
        # print("缩小后的宽度为:"+str(W))
        # print("缩小后的高度为:"+str(H))
        pix = self.curPixmap.scaled(W, H)
        self.ui.Pictures.setPixmap(pix)
        self.ui.Pictures.setAlignment(Qt.AlignVCenter)
        self.ui.Pictures.setAlignment(Qt.AlignHCenter)

    @pyqtSlot()
    def on_action_ZoomRealSize_triggered(self):
        self.pixRatio = 1
        self.ui.Pictures.setPixmap(self.curPixmap)
        self.ui.Pictures.setAlignment(Qt.AlignVCenter)
        self.ui.Pictures.setAlignment(Qt.AlignHCenter)

    @pyqtSlot()
    def on_action_adjustheight_triggered(self):
        H = self.ui.ScrollArea.height()  # 滚动区域的高度
        realH = self.curPixmap.height()  # 图片的高度
        self.pixRatio = float(H) / realH
        pix = self.curPixmap.scaledToHeight(H - 100)
        self.ui.Pictures.setPixmap(pix)
        self.ui.Pictures.setAlignment(Qt.AlignVCenter)
        self.ui.Pictures.setAlignment(Qt.AlignHCenter)

    @pyqtSlot()
    def on_action_adjustwidth_triggered(self):
        W = self.ui.ScrollArea.width()
        realW = self.curPixmap.width()
        self.pixRatio = float(W) / realW
        pix = self.curPixmap.scaledToWidth(W - 100)
        self.ui.Pictures.setPixmap(pix)
        self.ui.Pictures.setAlignment(Qt.AlignVCenter)
        self.ui.Pictures.setAlignment(Qt.AlignHCenter)

    '''设置停靠区浮动性、可见性'''
    @pyqtSlot(bool)
    def on_action_DockFloat_triggered(self, Checked):
        if Checked:
            self.ui.dockWidget.setFloating(False)
        if not Checked:
            self.ui.dockWidget.setFloating(True)

    @pyqtSlot(bool)
    def on_action_DockVisible_triggered(self, Checked):
        if Checked:
            self.ui.dockWidget.setVisible(False)
        else:
            self.ui.dockWidget.setVisible(True)
    '''停靠区浮动性、可见性改变'''
    @pyqtSlot(bool)
    def on_dockWidget_topLevelChanged(self, toplevel):
        '''
        if toplevel  :
            print("处于浮动状态")
        else:
            print("处于非浮动状态")
        '''
        self.ui.action_DockFloat.setChecked(toplevel)

    @pyqtSlot(bool)
    def on_dockWidget_visibilityChanged(self, visible):
        '''
        if visible :
            print("处于可见状态")
        else:
            print("处于不可见状态")
        '''
        self.ui.action_DockVisible.setChecked(visible)


app = QApplication(sys.argv)
bmp = QIcon(":/images/icons/29.ico")
form = MyTreeWidget()
form.setWindowIcon(bmp)
sys.exit(app.exec_())
