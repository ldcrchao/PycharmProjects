#%%
'''
一、以滑动条的动作信号为例分析 动作信号-信号内传递-信号跨类传递-信号处理-信号发射-信号继承-信号连接窗体-窗体组件状态改变 的传递过程
1、slider(滑动的动作)是不能直接传递到窗体其他组件的，所以需要将动作信号变为值的信号进行传递
2、slider的value需要传递给谁呢？文本组件。文本显示的信息需要根据value的变化作相应的处理，这就需要值处理函数
3、value本身需要传递外，value改变的信号也需要发射，这就需要一个值发射函数
4、值发射函数和值处理函数在human类中已经写好，但是自定义的业务逻辑类mywidget是和human分离的，所以首先需要建立2个类的联系
5、mywidget类调用human类实例化了1个对象boy作为mywidget的私有属性self.boy
6、mywidget类和human类的实例化对象self.boy如何建立联系？mywidget可以定义自己的槽函数,这里是on_setageslider_valueChanged函数
   不定义在__init__的self属性中,单独定义槽函数建立on_setageslider_valueChanged和self.boy.setAge的关系可以避免占用内存,
7、那么value -> on_setageslider_valueChanged(值内传递和跨类传递函数) -> self.boy.setAge(值处理函数) ->
   -> ageChanged(信号类的实例emit:value和value改变的信号,值发射函数) -> mywidget.boy.ageChanged(继承了ageChanged:
   因为值处理函数在human类中,不能在mywidget直接发射ageChanged,所以就需要间接改变和继承) ->
   -> do_ageChanged_int/str(connect连接了ageChanged信号和槽函数,内传递) -> self.ui(Ui_Human_Widget的GUI实例化对象) ->
   -> 窗体组件(只能通过mywidget的self.ui进行访问)
二、复选框的传递过程作为演练
复选框的勾选动作信号 -> on_checkBox_clicked(信号内部传递和跨类传递) -> self.boy.setName(信号处理) -> nameChanged(信号发射) ->
-> mywidget.boy.nameChanged(信号继承) -> do_nameChanged_str(信号连接窗体,内传递槽函数) -> self.ui -> 窗体组件
三、信号和槽函数的关联分为2种
1、内建信号clicked()和close(),如关闭按钮,只能通过窗体设计ui窗口本身的edit signals_slots 进行编辑,不能通过程序传递
   发送信号:"关闭"窗口(clicked) -> 接收信号:Human_Widget(close)
2、自定义信号,nameChanged和ageChanged,需要通过mywidget内置的槽函数执行self.ui的动作
四、槽函数参数的性质分为3种
1、C++函数原型中拷贝的函数名,“on”打头,例如on_setageslider_valueChanged
  将滑动条或者下拉条移动时值的变化赋给了函数,这是底层已经强制的参数,可以直接使用
2、C++函数原型也可以不带参数,例如按钮类是点击的动作信号,不需要传递信号,如on_checkBox_clicked
3、自定义的值处理函数或者__init__定义或继承的参数,参数属于自身定义的,如human的__init__定义的age可以通过mywidget实例化继承
   传递给boy,再通过boy.setAge(age)传递,因为传递了参数age,接收的函数do_ageChanged_int(age)也被强制拥有参数
4、被动强制接收参数的槽函数,如第3条提到的do_ageChanged_int(age),参数不是自定义的,可以接收后直接用
'''
import sys
from PyQt5.QtWidgets import  QApplication ,QWidget
from PyQt5.QtCore import  pyqtSlot
from PyQt5.QtGui import  QIcon
from GUI.Demo.Demo1 import Human_Widget as HW
from GUI import human
class QmyWidget(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.ui = HW.Ui_Human_Widget() #定义属性ui,指向Human_Widget窗体文件中的类Ui_Human_Widget
        self.ui.setupUi(self)

        # human是1个自定义的信号类,可以返回3个字符串输出,分别提示年龄、年龄段和称谓
        self.boy = human.Human("Boy",16) # 引用human的Human类实例化对象作为私有属性self.boy
        '''boy内部通过setAge和setName函数已经把参数age和name发射出去,只需要建立信号和槽函数的关联即可
        human文件的主程序是连接response类,输出3个信息,这里要求输出到窗体组件,通过自定义的3个槽函数和窗体组件关联'''
        self.boy.nameChanged.connect(self.do_nameChanged_str)
        #self.do_nameChanged_str("chenbei") # 这里说明了可以不使用nameChanged提供的信号和值,而是用自己定义的参数
        self.boy.ageChanged.connect(self.do_ageChanged_int)
        self.boy.ageChanged[str].connect(self.do_ageChanged_str)

    '''自定义的槽函数,设定好3个文本的输出'''
    def do_nameChanged_str(self,name):
        '''
        这里自定义的槽函数负责传递从boy改变的参数,boy(人对象)->nameChanged(名字改变的信号)->do_nameChanged_str(槽函数)
        ->self.ui.nametext(窗体文本对象)->setText(动作)
        '''
        self.ui.nametext.setText("Hello,"+name) #nametext是 nameChanged(str)响应的名字
    @pyqtSlot(int)
    def do_ageChanged_int(self,age):
        self.ui.agerestext.setText(str(age)) #将年龄传递到窗体年龄文本组件 ageChanged(int)响应
    @pyqtSlot(str)
    def do_ageChanged_str(self,info):
        self.ui.lineEdit_3.setText(info)  # 年龄段传递到ageChanged(str)组件

    '''槽函数,建立滑动条和信号的关联'''
    def on_setageslider_valueChanged(self,value):
        '''value是C++函数原型中就将滑动条值得变化与输入参数绑定,可以直接用'''
        self.boy.setAge(value) # 将滑动条移动时值改变的信号与setAge函数关联
        # self.boy虽然能够内部传递参数,但是是不可改变的,这里将滑动条直接关联setAge函数就可以实现改变,而不需要外部传递
    '''复选框与信号的关联'''
    def on_checkBox_clicked(self):
        checked = self.ui.checkBox.isChecked()
        if checked :
           hisname = self.ui.lineEdit.text() #选中复选框时,将左方名字的文本赋给hisname
           self.boy.setName(hisname)  # 输入姓名的窗口组件输入后,获取文本，发射该参数经过连接送到了槽函数,槽函数进行传递到窗体的工作
        else :
            self.ui.nametext.clear()
app = QApplication(sys.argv)
jpg = QIcon(":/jpgs/images/time.jpg")
app.setWindowIcon(jpg)
form = QmyWidget()
#form.ui.close.setGeometry(350,100,100,50)
form.show()
sys.exit(app.exec_())