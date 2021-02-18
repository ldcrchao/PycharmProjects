import sys
# 驼峰式命名法-多个单词首字母大写
# 方法  : 普通方法 、 类方法 、静态方法、 魔术方法
'''
1. __new__是在实例创建之前被调用的,因为它的任务就是创建实例然后返回该实例对象,是个静态方法
2. __init__是当实例对象创建完成后被调用的,然后设置对象属性的一些初始值,通常用在初始化一个类实例的时候,是一个实例方法
3. __new__先被调用,__init__后被调用,__new__的返回值（实例）将传递给__init__方法的第一个参数self,然后__init__给这个实例设置一些参数
4. 继承自object的新式类才有__new__
5. __new__至少要有一个参数cls，代表当前类，此参数在实例化时由Python解释器自动识别
6. __new__必须要有返回值，返回实例化出来的实例 参数self，就是这个__new__返回的实例 __init__不需要返回值
7. 可以return父类(通过super(当前类名, cls))__new__出来的实例，或者直接是object的__new__出来的实例
------>object.__new__(cls)  or super(Person, cls).__new__(cls)  一般用于单例模式
8. 单例模式 : 一个类有且仅有一个实例，并且自行实例化向整个系统提供  某种场景需要对象可以随意创建但是是同一个东西
9. 单例模式的要点有三个：一是某个类只能有一个实例；二是它必须自行创建这个实例；三是它必须自行向整个系统提供这个实例
'''
# 单例模式
class Person():
    __mode = None # 存放每次实例化的地址 且私有化属性可以防止外部更改
    def __new__(cls, *args, **kwargs):
        print("------------->调用__new__")
        if cls.__mode  is None :
            print('-------->1')
            #cls.__mode = object.__new__(cls) # 调用超类方法申请一个内存地址
            cls.__mode = super(Person, cls).__new__(cls)
            print(cls.__mode)# <__main__.Person object at 0x000001B67A765DD8>
            return cls.__mode  # 返回地址给__init__
        else:
            print('-------->2')
            return cls.__mode #
# 第一次调用 为空进入第一层 将地址给了__mode
# 由于没有 __init__ 所以__new__直接扔出地址
s1 = Person()
print(s1)
# 第二次调用 不为空进入第二层 将上一次地址给了__mode
# 两次返回的地址就会一致 不是开辟新空间
s2 = Person()
print(s2)
class Phone() :
    brand = 'huawei'
    number = 1 # 类属性 外界可以修改  直接访问 : Phone.number
    __number = 2 # __ 表示类的私有属性 外界不能修改 不能访问 : Phone.__number 但可以通过函数_cls___number()访问
    # 1.魔术方法 : 普通方法需要调用 , 魔术方法会自动触发
    # 1.1 : 初始化魔术方法__init__ , 初始化对象时触发(不是实例化触发,但是和实例化在一个操作)
    def __init__(self ,brand ,name) :
        print('初始化魔术方法--------->>>__init__运行到此位置')
        print("初始化魔术方法--------->>>__init__返回的地址为 : ",self)
        # 创建对象时就会默认执行的函数
        # 这里的brand(对象属性)和上方的brand(类属性)不是一个东西
        self.brand = brand
        self.name = name
    # 1.2 : 实例化魔术方法__new__ , 申请内存地址 , 返回的是一个地址,然后才执行__init__初始化,这个地址也是phone1的地址
    def __new__(cls, *args, **kwargs):
        print('实例化魔术方法--------->>>__new__运行到此位置')
        print("实例化魔术方法--------->>>__new__返回的地址为 : ",object.__new__(cls))
        return  object.__new__(cls)
    # 1.3 : 调用魔术方法__call__ , 可以将复杂的步骤进行合并操作 , 减少调用的步骤方便使用
    # 想要把对象当成函数使用 可以使用__call__重写
    def __call__(self, name):
        '''
        :arg : 至少一个self接收对象,其余根据调用时参数决定
        :return : 根据情况而定
        '''
        print('调用魔术方法--------->>>__call__运行到此位置')
        print('调用魔术方法--------->>>__call__运行后的结果为 : ',name)
    # 1.4 : 析构魔术方法__del__ , 删除一次引用时执行的函数
    def __del__(self) :
        print('析构魔术方法--------->>>__del__运行到此位置')
    # 1.5 : 字符魔术方法__str__ , 返回对象名称而不是地址
    # def __str__(self):
    #     print('字符魔术方法--------->>>__str__运行到此位置')
    #     return self.name # 与__new__冲突了

    # 2.对象方法 : 依赖于self 不依赖cls , 只能对象调用 调用格式 ---> < phone1.brand >
    def _self_brand(self) :
        print('---->依赖于self不依赖于cls的对象方法 : ',self.brand)

    # 3.类方法 : 不依赖于self 依赖于cls , 类和对象都可以调用 调用格式 ---> < Phone.brand >
    @classmethod
    def _cls_brand(cls): # 这里cls  <---> Phone
        print('---->依赖于cls不依赖于self的类方法 : ',cls.brand)
        # 类方法内部不能调用对象方法
        # _self_brand() Error
    @classmethod
    def _cls___number(cls):
        print("---->依赖于cls不依赖于self的类方法 : ",cls.__number)

    # 4.静态方法 : 不依赖于self且不依赖于cls   对象和类都可以调用 调用格式 ---> < Phone.brand >
    @staticmethod
    def _stat_brand():
        # print(self.brand) # Error
        # print(cls.brand) # Error
        print('---->不依赖于self和cls的静态方法 : ',Phone.brand)


# 类
Phone.number = Phone.number + 1 # 可以修改
print("直接访问可以修改的内部类属性---> ",Phone.number ) # output : 2
# Phone.__brand = Phone.__number +  1 # 不可以修改
print("间接访问不可以修改的内部类属性---> ",Phone._cls___number())

# 对象
phone1 = Phone('hua','cb') # 1次引用
# print(phone1) # <__main__.Phone object at 0x000001A937547D30>
# 可以发现__init__/__new__/phone1的地址完全相同
print('创建对象--------->>>Phone()运行到此位置')
print("创建对象--------->>>Phone()返回的地址为 : ", phone1)

# 方法
# 1.魔术方法
# 1.1 初始化方法__init__ 创建对象时即自动执行 初始化对象时触发
# 1.2 实例化方法__new__ 优先于__init__执行,用于申请内存返回地址 实例化时触发
# 1.3 调用方法__call__ 将创建的对象当成函数使用时触发 自动跳到__call__函数,重写的函数必须在__call__内部
phone1("chenbei")
# 1.4 析构方法__del__ 当对象没有被引用时触发 触发垃圾回收机制释放内存 函数了解即可不需要自定义
# 返回某个变量被引用的次数
p1 = phone1 # 2次引用
p2 = phone1 # 3次引用
print("实例化对象phone1被引用的次数为(包括本次) : ",sys.getrefcount(phone1)) # 4次引用
del p1
# print("删除引用p1后的输出为 : ",p1.name) # NameError: name 'p1' is not defined
del p2
print("实例化对象phone1被引用的次数为(包括本次) : ",sys.getrefcount(phone1)) # 2次引用
# del phone1 # 删除唯一的引用时触发__del__函数 # output : 析构魔术方法--------->>>__del__运行到此位置
# 1.5 字符魔术方法__str__ , 返回对象名称而不是地址 与__new__冲突了单独使用例子说明
# __str__可以打印自己想要的字符 返回的实例化对象不再是地址而是想要的字符串
class person():
    __id = 10
    def __init__(self,name, age ):
        self.__name = name
        self.__age = age
    # 用于直接获取私有属性 先有get动作
    @property
    def age(self):
        return self.__age
    # 用于直接改变私有属性 再有set动作 set依赖于get
    @age.setter
    def age(self , age):
        self.__age = age
    def setName(self,name) :
        self.__name = name
    def getName(self) :
        return self.__name
    def __str__(self):
        return self.__name
pa  = person('test',18)
print("返回的是pa的变量而不是地址 : ",pa) # 返回的不是地址而是变量值
# print(dir(pa)) # 可以得到底层的函数和变量变式
# print(pa.__name) # 直接访问私有属性不能访问是因为底层将其改名 变为_person__name
'''1.私有属性__name和__age可以通过定义函数间接获取和访问'''
# print(pa._person__name)# _类名__私有属性名  test 伪私有属性 仍然可以访问
pa.setName('chenbei')
name = pa.getName()
print(name)
'''2.也可以使用装饰器@property访问 外界就可以直接获取属性 但是函数和变量必须同名'''
# 用于直接获取私有属性
print(pa.age) # 不需要加() 即pa.age()
# 直接改变私有属性
pa.age = 100
print(pa.age)

# 2.对象方法(普通方法)
# 对象可以调用对象方法 但是类不可以调用对象方法
phone1._self_brand() # output : ---->依赖于self不依赖于cls的对象方法 :  hua
# Phone._self_brand() Error

# 3.类方法 : 对象创建之前需要完成一些动作则可以使用类方法
# 类可以调用类方法 对象也能调用类方法
phone1._cls_brand() # output : ---->依赖于cls不依赖于self的类方法 :  huawei
Phone._cls_brand()  # output : ---->依赖于cls不依赖于self的类方法 :  huawei

# 4.静态方法 : 类和对象都可以调用静态方法 , 非对象实例也能使用该方法
Phone._stat_brand() # output : ---->不依赖于self和cls的静态方法 :  huawei
phone1._stat_brand() # output : ---->不依赖于self和cls的静态方法 :  huawei

# 内部的对象属性
print("内部的对象属性---> ",phone1.brand) # output : hua
# 内部的类属性
print("内部的类属性---> ",Phone.brand ) # output : huawei
# 外部动态 新定义对象属性 //不是公共特征 一般不使用
phone1.note = "我是外部程序动态添加的对象属性"
print("外部动态新定义对象属性---> ",phone1.note)
# 外部动态 新定义类属性 //不是公共特征 一般不使用
Phone.note = "我是外部程序动态添加的类属性"
print("外部动态新定义类属性---> ",Phone.note)
