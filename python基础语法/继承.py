# 关联关系
# 继承关系之前的为关联关系 即多个类为平行关系
# class Book ;class Phone; class Person ;
# book = Book() ; phone = Phone(); person = Person()
# 在Person的__inint__函数中可以定义(self,book,phone)
# 书本类实例化和手机类实例化可以作为人类的两个形参
# 然后在送入实参person = Person(book,phone)

# 继承关系
# 1. 子类中如果不定义__init__则调用父类的super class的__init__
# 2.若子类定义了__init__ 需要在内部调用一下父类__init__
# 3.两种方式调用父类
# 3.1 def __init__(self, arg1,arg2,...) arg1,arg2,..都是父类的init参数
#       super().__init__(arg1,arg2,...)
# 3.2 def __init__(self,arg1,arg2)
#       super(子类名 , self).__init__(arg1,arg2,...)
# 4.子类方法和父类方法有一个同名函数 则先查找子类再去找父类
# 5.由于4.的查找关系可以看出子类方法可以重写父类方法
# 6.子类方法调用父类方法 使用格式为  :
# 6.1 self.father()
# 6.2super().方法名(参数) 

# 多重继承
import inspect
'''
继承关系: 从左到右 深度优先
          A
        / | \
       /  |  \
      /   |   \
     B    C    D
     \    /\  /
        E    F
         \  /
           G
0、G的内容优先级最高                                                                G
1、G当前子类，查找父类，G的第一父类E不是最后1个指向G的，还有F，则记录E，且继续查找E的父类    E
2、E当前子类，查找父类，E的第一父类B不是最后一个指向E的，还有C，则记录B，且继续查找B的父类   B
3、B当前子类，查找父类，B只有1个父类A，则跳过，返回G的分支F类，记录F，且继续查找F的父类      F
4、F当前子类，查找父类，F的第一父类C不是最后一个指向F的，还有D，则记录C，且继续查找C的父类    C
5、C当前子类，查找父类，C只有1个父类A，则跳过，返回F的分支D类，记录D，且继续查找D的父类      D
6、D当前子类，查找父类，D只有1个父类A，且无其它分支，记录A                               A
'''
class A():
    def __init__(self):
        print('A')
class B(A):
    def __init__(self):
        print('B')
        super().__init__()
class C(A):
    def __init__(self):
        print('C')
        super().__init__()
class D(A):
    def __init__(self):
        print('D')
        super().__init__()
class E(B,C):
    def __init__(self):
        print('E')
        super().__init__()
class F(C,D):
    def __init__(self):
        print('F')
        super().__init__()
class G(E,F):
    def __init__(self):
        print('G')
        super().__init__()
G()
print(inspect.getmro(G))
print(G.__mro__)