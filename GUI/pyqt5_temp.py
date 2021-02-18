'''缺陷:面向过程的应用程序框架,只适合测试单个窗体的UI设计效果
一般的业务逻辑是从界面上读取数据,处理后再将结果输出到界面上
这里baseWidget只是1个QMainWindow类的对象,是为了能够访问setupUI函数而传入的一个参数,不包含任何业务处理逻辑
基于以上缺陷,还需要界面与逻辑分离的GUI程序框架,方法主要有2种:单继承法和多继承法
'''

'''面向过程的GUI框架'''
import sys
from PyQt5 import QtWidgets,QtGui
from GUI import ui_HelloWorld
app = QtWidgets.QApplication(sys.argv)
basewidget = QtWidgets.QMainWindow()
ui = ui_HelloWorld.Ui_PrintHello() # GUI框架文件定义了1个QMainWindow类,需要一个QMainWindow参数去实例化
ui.setupUi(basewidget)  # 实例对象还没有传递QMainWindow参数,这里传递
basewidget.setWindowTitle("Hello World!") # 可以通过访问窗体本身的属性 , 但是不能访问窗体上的组件属性 , 只能通过ui访问,ui定义了这些公共属性
#basewidget.resize(1200,1000)
basewidget.show() # 不可以使用ui.show(),basewidget是实际的窗体参数,ui是一个实例化对象但是没有参数的空壳
#ui.LabHello.setText("陈北最帅!") # 可以通过ui去访问窗体类的下级对象
#font = QtGui.QFont() # 格式类
#font.setBold(True) #加粗
#font.setPointSize(20) # 字体大小
#font.setStyleName("Time New Roman") # 字体风格
#ui.LabHello.setFont(font) #传递字体类对象
#size = ui.LabHello.sizeHint()  # 获取对象最适合的大小
#ui.LabHello.setGeometry(100,700,size.width(),size.height()) # 传递宽度、高度参数
sys.exit(app.exec_())

#%%
'''
逻辑与界面分离的GUI框架
多继承法:优点在于公共属性可以直接访问.访问方便；缺点是过于开放,新定义的属性和公共属性不易区分。
'''
import sys
from PyQt5.QtWidgets import  QMainWindow  ,QApplication
from GUI import ui_HelloWorld
class MyQMainWindow(QMainWindow,ui_HelloWorld.Ui_PrintHello):
    '''
    自定义的类继承了Qt的QMainWindow类和GUI框架中的QMainWindow类
    前者是为了使用该类的方法,后者是为了使用该类的框架
    MyQMainWindow与多继承的第1个类是一致的,所以相当于自定义的类是一个Qt下的QMainWindow对象,而且还具备了其它基类的属性和方法
    逻辑关系:
    MyQMainWindow(自定义类):
        继承:
        ->    QMainWindow(顶级父类1)(self)
                  ->    顶级父类1的方法 : 如setWindowTitle、resize,show等
                  ->    顶级父类1的属性
                  ->    次级父类1
                            ->    次级父类1的方法
                            ->    次级父类1的属性
                            ->    次次级父类1
                                  ...
                            ->    次次级父类n
                  ->    次级父类2
                  ->    ...
                  ->    次级父类n
        ->    Ui_PrintHello(顶级父类2)
                  ->    顶级父类2的方法 : setupUi
                  ->    顶级父类2的属性 : LabHello, close , Start
                  ->    次级父类1
                            ->    次级父类1的方法
                            ->    次级父类1的属性
                            ->    次次级父类1
                                  ...
                            ->    次次级父类n
        自定义:
        ->    顶级父类3
                  -> 顶级父类3的方法
                  -> 顶级父类3的属性
                  -> 次级父类1
                  ...
                  -> 次级父类n
        ->    ...
        ->    顶级父类n
        ->    方法1
        ->    ...
        ->    方法n
        ->    属性1
        ->    ...
        ->    属性n
    '''
    def __init__(self,parent=None): #parent是父类的参数
        super().__init__(parent) # 继承父类QMainWindow,Ui_PrintHello
        # 执行此语句后self变为Qt下的QMainWindow类的1个对象 , 这里多继承时第1个基类成为self
        self.Lab = "多重继承的MyQmainWindow" #自定义独有特性
        self.setupUi(self)  # 这里self自身是个QMainWindow对象,但是也继承了PrintHello基类的属性和方法,可以直接使用setupUi方法
        self.LabHello.setText(self.Lab) # 直接使用第2个基类的属性,传递的是自定义的属性
# 主程序
app = QApplication(sys.argv)
MyQMainWindow = MyQMainWindow()
MyQMainWindow.show() # 直接使用第1个基类的方法
MyQMainWindow.Start.setText("A Great Start Again!")  # 也可以外部程序直接使用第2个基类的属性及其方法
MyQMainWindow.close.setText("关闭")
MyQMainWindow.setWindowTitle("Hello World!")
sys.exit(app.exec_())

#%%
'''单继承法'''
import sys
from PyQt5.QtWidgets  import QMainWindow  ,QApplication
from GUI import ui_HelloWorld
class MyQMainWindow(QMainWindow):#只继承Qt下的QMainWindow
    '''
    可视化设计的窗体对象被定义为MyQMainWindow的一个私有属性self._ui，外界访问窗体对象只能通过私有属性访问，而不能直接访问
    MyQMainWindow的自定义属性self.Lab不会与窗体组件混淆,self._ui.LabHello.setText(self.Lab) and self.LabHello.setText(self.Lab)
    '''
    def __init__(self,parent=None):
         super().__init__(parent)
         #self._ui = ui_HelloWorld.Ui_PrintHello() # 因为没有继承PrintHello类,可以通过直接使用该类定义为私有属性(该属性为类)
         # 相对于多继承而言至少PrintHello类不会暴露,只能通过self._ui的属性访问界面组件,而非self的属性去访问
         #self._ui.setupUi(self)  # 使用该类的方法继续定义新属性,构造UI, 框架需要传递QMainWindow参数,即self
         self.Lab = "单继承法的GUI框架"
         #self._ui.LabHello.setText(self.Lab)

         self.ui = ui_HelloWorld.Ui_PrintHello()  # 公有属性
         self.ui.setupUi(self)

    def setbuttontext(self,text):
        #self._ui.close.setText(text)
        self.ui.close.setText(text)
# 主程序
app = QApplication(sys.argv)
MyQMainWindow = MyQMainWindow()
MyQMainWindow.show()
MyQMainWindow.setbuttontext("点击")
MyQMainWindow.setWindowTitle("Hello World!")
#MyQMainWindow._ui.Start.setText("Another Great Start!") # 通过_ui属性间接访问,多继承可以直接访问
#MyQMainWindow._ui.close.setText("间接关闭")
MyQMainWindow.ui.close.setText("间接关闭")
sys.exit(app.exec_())
#%%
'''MyDialog，新建了一个完整的项目，对话框类'''
import sys
from PyQt5.QtWidgets import  QApplication ,QDialog #对话框类
from GUI import HelloWorld1
from PyQt5.QtCore import  Qt , pyqtSlot
from PyQt5.QtGui import  QPalette
class MyDialog(QDialog):
    '''
    理解为自定义了一个对话框类，不仅继承了QDialog，而且定义了自己的私有属性
    只是该属性继承了已经设计好的Dialog窗体GUI框架
    这样自己定义的类不仅可以使用QDialog的方法，也可以使用ui_NormalDialog的框架
    self就是1个Dialog类，所以可以使用QDialog，而设计好的ui_NormalDialog中的类也是Dialog类
    而self.ui已经被定义为Dialog类,所以self.ui也可以使用Ui_NormalDialog
    '''
    def __init__(self,parent=None):
        # 和super(Human,self).__init__(parent)是完全一样的
        # parent是父类的一个参数,如果选择了parent表明子类继承了父类的参数,super()用于初始化是否继承属性
        '''
        父类person具有属性name
        class person():
              def __init__(self,name='chenbei',address='BeiJing'):
                  self.name = name
                  self.address = address
        子类son_one不做初始化则会继承父类person的属性name
        class son_one(person):
              pass
        子类son_two做了初始化且建立了自己的属性,但是不调用super继承person的name属性
        class son_two(person):
              def __init__(self,age):
                  self.age = age
        子类son_three初始化建立自己属性,且调用super继承person的属性
        class son_three(person):
              def __init__(self,age,name):  #如果希望允许外部程序修改name属性,需要在这里添加name参数
                  self.age = age
                  super().__init__(name)
        class son_four(person):
              def __init__(self,age): # 如果不指定参数,则意味着继承的参数是不允许修改的
                  self.age = age
                  #super(son_four, self).__init__()
                  super().__init__()
        '''
        super().__init__(parent)
        #self.ui = ui_NormalDialog.Ui_NormalDialog() #使用的是TextEdit对象,按钮有一些无法使用
        self.ui = HelloWorld1.Ui_NormalDialog()   # PlainTextEdit对象
        self.ui.setupUi(self)
        self.ui.Black.clicked.connect(self.change_textcolor) # 手动关联信号和槽函数
        self.ui.Blue.clicked.connect(self.change_textcolor)
        self.ui.Red.clicked.connect(self.change_textcolor)
    '''以下方法都是手动在python程序中直接编写,自动关联信号和槽函数'''
    def on_Clear_clicked(self):
        '''在Qt文件下右击清空按钮，关联到槽，选择clicked()信号，可以看到相应的C++文件中多了槽函数的定义
        然后将其自动生成的函数名字复制，在自定义的类写该函数即可,此名字必须使用对应C++函数的格式，否则无法实现函数和信号关联
        关闭和确定两个按钮的槽函数自带，不需要单独创建函数'''
        self.ui.plainTextEdit.clear() # 将清空按钮的clicked信号关联到清空文字的命令
    def on_Bold_toggled(self,checked):
        '''这里的区别在于QCheckBox类型需要关联的槽函数是toggled(bool)函数,表示在复选框的状态变化时发射信号,将复选框的勾选状态作为参数进行传递
        同时多了一个参数checked，源C++程序带有该参数，这也是复选框的特别之处
        除此之外还有clicked(bool)信号,clicked()信号是点击就会传递,而clicked(bool)是点击复选框时的勾选状态作为一个参数传递
        这两个名称相同但是参数个数和类型不同的信号也叫overload型信号
        toggled(bool)和clicked(bool)的区别见下划线函数
        '''
        font = self.ui.plainTextEdit.font() #实例化文本标签的格式类
        font.setBold(checked) # 表示选中状态才会执行,checked本身就是个bool值类型
        self.ui.plainTextEdit.setFont(font)
    def on_Underline_clicked(self):
        '''在ui_NormalDialog中已经设置了underline复选框的初始状态是勾选的
        但是文字并没有出现下划线，取消复选框，文字不出现下划线，再次选中才出现下划线'''
        checked = self.ui.Underline.isChecked() #读取勾选状态
        font = self.ui.plainTextEdit.font()
        font.setUnderline(checked)
        self.ui.plainTextEdit.setFont(font)

    @pyqtSlot(bool)
    def on_Italic_clicked(self,checked):
        font = self.ui.plainTextEdit.font()
        font.setItalic(checked)
        self.ui.plainTextEdit.setFont(font)
    '''
    def on_Italic_toggled(self,checked):
        font = self.ui.plainTextEdit.font()
        font.setItalic(checked)
        self.ui.plainTextEdit.setFont(font)
    '''
    '''
    # 可以单个进行自动关联，也可以合并在一起写，但是需要手动建立信号和槽函数的关系
    def on_Red_clicked(self):
        plet = self.ui.plainTextEdit.palette()
        checked = self.ui.Red.isChecked()
        if checked :
           plet.setColor(QPalette.Text,Qt.red)
        self.ui.plainTextEdit.setPalette(plet)
    '''
    '''手动关联信号和槽函数'''
    def change_textcolor(self):
        plet = self.ui.plainTextEdit.palette() #实例化调色板
        if (self.ui.Red.isChecked()):
            plet.setColor(QPalette.Text,Qt.red) #按钮被选中时给对应按钮的文字被设定相应的颜色
        elif (self.ui.Blue.isChecked()):
            plet.setColor(QPalette.Text,Qt.blue)
        elif (self.ui.Black.isChecked()):
            plet.setColor(QPalette.Text, Qt.black)
        self.ui.plainTextEdit.setPalette(plet)

app = QApplication(sys.argv)
mainform = MyDialog()
#plet = mainform.ui.plainTextEdit.palette() # 外部程序无法更改响应
#if  mainform.ui.Red.isChecked():
#    plet.setColor(QPalette.Text, Qt.green)
#    mainform.ui.Red.setPalette(plet)
mainform.show()
sys.exit(app.exec_())
