# 闭包概念:函数定义了内部函数,函数的返回值是内部函数
'''
def 外部函数():
    ...
    def 内部函数():
        ...
    return 内部函数
'''
def a(args):
    a = 100
    def b(name):
        print('内部函数的输出变量为名字{},外部的输出变量为数字{}'.format(name,args)) # 这里内部函数可以不需要单独定义直接引用外部函数的变量
    #print(locals())
    return a,b
a1,b1 = a(10) # 内部函数也可以作为返回值,即闭包操作
b1('chenbei') # 内部函数的输出变量为名字chenbei,外部的输出变量为数字10
a2,b2 = a(5)
b2('chenbei')
# out: 内部函数的输出变量为名字chenbei,外部的输出变量为数字5
b1('chenbei') # 可以看出即使a()变量改变为5,但是重复调用b1函数输出仍然是10
# 或者说b1,b2记录了当时传入的外部参数状态,两个函数是不同的,它们的抛出值内存地址不相同
# out: 内部函数的输出变量为名字chenbei,外部的输出变量为数字10

# 闭包的应用计数器
def generate_count() :
    count = [0]
    def add_one() :
        count[0] = count[0] + 1
        print('当前是第%d次访问' % count[0])
    return add_one
counter = generate_count()
counter()

# 装饰器概念:函数作为参数
def ifunc(number):
    a = 100
    def inner_ifunc() :
        #global a # a不是全局变量,不能使用global声明
        nonlocal a # nonlocal声明的变量不是局部变量,也不是全局变量,而是外部嵌套函数内的变量
        for i in range(number) :
            a = a + 1 # 对内部函数而言a是不可变的全局变量 不能修改
        print(a)
    return inner_ifunc
def ifuncf(f) :
    f()
f = ifunc(5)
ifuncf(f) # 函数作为参数传递

# 装饰器的应用:某个场景有个函数在定义时考虑的不全面,之后还需要增加一些功能
# 有两个方法,第一个改变现有函数,这可能导致其它用到该函数的也需要改变
# 第二方法增加新函数,在新函数中调用原有函数 完善好修饰的代码,这样会导致代码量大时原有函数被替换为新函数的工作变得复杂
# 最后一个方法就是装饰器,定义某个函数作为变量的函数,即可以用@新函数 装饰原有函数
def decorate(f) :
    a = 10
    #print('第一次装饰器启动前')
    def wrapper(*args,**kwargs): # 此函数其实就接受了add函数的参数值,没有改变原有add函数通过wrapper()输出新功能的add函数
        ''':args 因为其他函数也可能用到这个装饰器,对add函数而言用了4个参数,如果不使用*args
        而使用wrapper(a,b,c,d)会导致别的函数调用该装饰器出错,而反复更改需求,所以最好使用*args作为万能装饰器'''
        # print(*args) # 1,2,3,4
        # print(args) # (1,2,3,4)
        sum , list = f(*args,**kwargs) # add函数是对常数类型处理的,需要使用*args而不是args元组类型
        # add函数的*args会将零散的args装包成元组,所以不能直接输入已经装包的元组
        # 关键字参数和字典参数也需要先使用**变量名解包,然后函数内部会自行再装包对应
        sum = sum + a
        print('第一次装饰后的结果为------->\nsum = ',sum)
        print(list)
        return sum, list # 如果这里不返回参数,由于第二次装饰器的输入函数是第一次装饰器的函数,再调用会报错没有返回值
    #print('第一次装饰器加载完毕')
    return wrapper
def Decorate(f) :
    #print('第二次装饰器启动前')
    def wrapper(*args,**kwargs) :
        print("开始调用第一层装饰器.....")
        #print(args,kwargs)
        sum,list = f(*args,**kwargs) # 这里的wrapper函数不再是add()函数,而是第一次装饰后的wrapper()函数
        # 所以这个函数得有返回值才能接收
        print("第二次装饰后的结果为------->")
        sum = sum + 50
        print('sum is {} '.format(sum))
        return sum
    #print('第二次装饰器加载完毕')
    return wrapper

@Decorate # 可以多次装修,在第一次装修的基础上
@decorate # 返回wrapper()函数,即叠加了add()函数的wrapper()函数
def add(*args,**kwargs):
    sum = 0
    list = []
    for i in args :
        sum += i
    for i in kwargs.items() :
        #print(i[0])
        #print(i[1])
        list.append(i[1])
    return sum ,list
#add(1,2,3,4) # 此时add()不再是add()了,而是wrapper(),所以如果wrapper()不提供接收的参数会报错
person={'name':'cb','sex':'boy','birth':199791}
add(1,2,3,4,*[10,20],**person,hobby='basketball')
#%%
# 装饰器的应用
import time
islogin = False
#
def login():
    username = input('输入用户名:')
    password = input('输入密码:')
    if username == 'admin' and password == '123456' :
        return True
    else:
        return False
# 定义装饰器修改登陆状态
def islogin_request(func): # 输入参数为函数
    def wrapper(*args,**kwargs): # 执行pay函数后wrapper变成了pay函数
        global  islogin  # 修改登陆状态
        # 调用pay函数后wrapper变成pay函数的加强版
        # 验证用户是否登录
        print('-----准备付款-----')
        if islogin :
           func(*args,**kwargs) # 默认是没登陆,所以先执行登陆界面
        else:
            # 跳转登陆界面
            print('用户还未进行登录,请执行登陆操作')
            islogin = login() # 调用函数执行登录操作,输入用户名和密码,相等时改变登陆状态为True
            while not islogin :
                print('用户登录失败,请重新登录')
                islogin = login()
            else:
                print('用户登录成功！')
                func(*args, **kwargs)  # 这里需要加入该代码,用户登陆成功后直接执行付款操作,不比重复进行第二次付款
                # 第一次会拦截,没有付款成功,先执行登录操作,成功后再次付款才能成功
    return wrapper

@islogin_request # 装饰器
def pay(money) :
    print('正在付款,付款金额是{}元'.format(money))
    print('付款中....')
    time.sleep(2)
    print('付款成功！')
pay(500)