# 单星号可变参数为元组
# 其它参数name必须放在*args前边,类似于a,*b=1,2,3,4,那么a=(1,),b=(2,3,4)
def add(name,*args) :
    sum = 0
    #print(name)
    # 传递多参数时,实际上args属于元组类
    if isinstance(args,int) or isinstance(args,float) or isinstance(args,tuple):
       for i in args :
           sum += i
       print('%s的求和成功为%f'%(name,sum))
    else:
        print('您的输入数据格式有误')
add('cb',1,2,3)
# add(name='cb',1,2,3) # 单星号是元组,不能有关键字参数,否则解包也会失败
# add(1,2,3,'cb') # 单星号参数必须放后头,否则参数解包出现问题 , 即会将name=1
# add(1,2,3,name= 'cb') # 提示add函数没有参数 name
def print_score(name,**args) :
    ''':type:**变量名 在底层会拆成字典形式
    或者直接输入关键字参数key=value'''
    if isinstance(args,dict) :
       for i in args.items() :
           # print(type(i))  # 字典拆成多个元组
           print('{}的{}成绩为{}'.format(name,i[0],i[1]))
    else:
        print('您的输入数据格式有误')
sum_score = {'math':79,'chinese':98,'english':60}
# print_score('cb',sum_score) #提示多给了参数
# print_score(name='cb',sum_score) # 提示应当给定关键字参数
# print_score(name='cb',**sum_score)  # 正确
print_score('cb',**sum_score) # 不提供关键字参数也可
# print_score('cb',**sum_score,a=1) # 会输出cb的a成绩为1,这说明a=1与**sum_score合并了,并覆盖了第1个字典元素
#%%
def print_args(a,b,*c,**d):
    print(a,b,c,d)
# print_args(1,2) # 1 2 () {}
# print_args(1,2,3) # 1 2 (3,) {} 不使用关键字参数,按顺序给定
# print_args(1,2,3,4,5) # 1 2 (3, 4, 5) {} 如果传入的不是关键字参数,解包时在满足必须参数后全部给*c参数
# print_args(1,2,3,4,5,c=6,d=7) # 1 2 (3, 4, 5) {'c': 6, 'd': 7} 关键字参数都给**d参数
# print_args(x=1,y=2) # 提示确实必须参数 a,b
# print_args(a=1,b=2) # 1 2 () {} # 使用关键字参数,按顺序给定
# print_args(a=1,b=2,3) # 提示位置参数只允许关键字参数
# print_args(a=1,b=2,c=3) # 然而c=3不作为元组参数,还是作为字典参数
# print_args(a=1,b=2,c=(3,4,5)) # 1 2 () {'c': (3, 4, 5)} 这里c=(3,4,5)仍然作为字典参数不作为元组参数
# print_args(1,2,*[3,4]) # 1 2 (3, 4) {} 不使用关键字参数 列表可以拆包
# print_args(a=1,b=2,*[3,4]) # 提示a有多个值
print_args(1,2,*[3,4]) # 故*变量名 能够传递的前提是前边不能有关键字参数

# 全局参数和局部参数
name = 'cb'
list = [1,2,3]
def loc_args():
    name = 1
    print(name)
def global_args():
    #name = name + '1997' # 全局变量不允许直接修改
    global name
    name = name + '1997' # 不可变对象事先声明后可以修改全局变量
    print(name)
    list.append(4) # 对于可变对象可以不声明直接修改
loc_args() # 局部变量
global_args() # 全局变量
print(list)