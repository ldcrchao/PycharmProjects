#%%
# -*- coding: utf-8 -*-
'''
自定义信号的参考示例：
自定义信号的使用特点：
1、一个信号可以关联多个槽函数，一个槽函数也可以与多个信号关联
2、信号之间也可以互相关联
3、信号的参数可以是任何python数据类型
4、关联可以直接(同步),也可以排队(异步)
5、可以在不同线程建立关联
6、信号和槽也可以断开连接
7、自定义信号是通过PyQt5.QtCore.pyqtSignal()完成的,且继承了父类QObject
'''
from PyQt5.QtCore import  pyqtSignal,pyqtSlot ,QObject
class Human(QObject):
    nameChanged = pyqtSignal(str) # 定义带str参数的信号
    ageChanged = pyqtSignal([int],[str]) # 定义带2个参数的信号 ,overload型信号有2个参数, int or str
    def __init__(self,name='Mike',age=10,parent=None):
        super().__init__(parent) # 和super(Human,self).__init__(parent)是完全一样的
        self.setAge(age)
        self.setName(name)
        self.resp = Responsor() # 必须实例化,self.nameChanged.connect(Responser.do_nameChanged_str)不能成功
        self.nameChanged.connect(self.resp.do_nameChanged_str) # 可以在主程序中进行连接外部函数,也可以在内部就连接好
        self.nameChanged.connect(self.handsomename)

    def handsomename(self,name): # 还可以在内部定义响应函数,连接内部函数
        print(name+",you are so handsome!\n")

    def setAge(self,age):
        #self._age = age  # 当使用setAge函数时就会给self添加属性,不在__init__直接定义可以不需要使用时少占用资源
        # 调用self的信号类
        self.ageChanged.emit(age) # 能使用self._age,或者age
        # overload信号默认发射第1个参数,即int信号
        if age<=18:
            ageInfo = "少年"
        elif (18<age<=35):
            ageInfo = "青年"
        elif (35<age<=60):
            ageInfo = "中年"
        elif (age>60):
            ageInfo = "老年"
        self.ageChanged[str].emit(ageInfo) #需要指定信号时必须先索引对应的信号

    def setName(self,name):
        self._name = name
        self.nameChanged.emit(self._name)
        #print(temp)
class Responsor(QObject):
    @pyqtSlot(int)
    def do_ageChanged_int(self,age):
        print("你的年龄是:"+str(age))
    @pyqtSlot(str)
    def do_ageChanged_str(self,ageInfo):
        print("你属于:"+ageInfo)
    def do_nameChanged_str(self,name):
        print("Hello,"+name)
if __name__ == '__main__':
    print("*****创建对象*****")
    boy = Human("Boy", 88)
    resp = Responsor()
    boy.nameChanged.connect(resp.do_nameChanged_str)  # 让内部实例化的'名字'信号与外部函数'修改名字'建立连接

    boy.ageChanged.connect(resp.do_ageChanged_int)  # 默认使用第1个参数,所以不需要额外指定[int]
    boy.ageChanged[str].connect(resp.do_ageChanged_str)  # 2个类的实例之间进行连接

    print("*****连接已经建立*****")
    # 使用boy类的2个方法,一旦启用,方法内部会将相应的信号发送出去
    # setAge(age) ->ageChanged(age改变的信号) -> 检测到后传递给do_ageChanged_int函数
    # 检测到age改变后接收被发送的参数age,该函数进行反馈,如输出对应信息
    boy.setName("Jack")
    boy.setAge(88)

    print("\n*****准备断开连接*****")
    boy.ageChanged[str].disconnect(resp.do_ageChanged_str)  # 断开改变年龄段的连接
    print("\n*****断开改变年龄段的关联后*****")
    boy.setAge(10)
    print("\n*****继续断开改变年龄的关联后*****")
    boy.ageChanged[int].disconnect(resp.do_ageChanged_int)  # 断开改变年龄的连接
    temp = boy.setAge(10)
    if temp == None:
        print("改变年龄和年龄段的连接已全部断开,不返回值")
    print("\n*****现在开始断开改变名称的关联*****")
    boy.nameChanged.disconnect(resp.do_nameChanged_str)
    temp1 = boy.setName("chenbei")
    if temp1 == None:
        print("改变名称的连接已断开,不返回值")

