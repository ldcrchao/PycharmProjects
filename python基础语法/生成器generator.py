# 
#1. g = (表达式 for 变量 in 列表)
#2. 函数 + yield
'''def func()
      ....
      yield
 g = func()
'''
# 系统函数 next(g) / 迭代器方法 __next__()


#可迭代对象:1.生成器2.列表、元组、元组、字典、字符串
# 可迭代对象从集合的第一个元素开始访问直到所有元素结束
# 可迭代对象不一定是迭代器,例如列表并不能使用next()函数,迭代器只能向前不能退后
# 如果希望把可迭代的变成迭代器,可以利用iter(iterable)函数
from  collections import Iterable
a = [1,2]
print(isinstance(a,Iterable))
b = iter(a)
print(next(b))

# 生成器generator 属于可迭代的对象,也是迭代器
# 列表推导式需要遍历所有元素,由于容量和内存有限
# 希望能够以某种算法只得到或改变其中的有限值
# 即总是可以由前边的元素推导到后边的元素,这样的表达式叫做生成器 不必创建完整列表

# 1.通过列表推导式得到生成器
list = [x*x  for x in range(20) ] # 生成20元素的列表

'''利用列表表达式得到生成器'''
# 希望只用到前几个元素 得到生成器
generator = (x*x  for x in range(20))
# print(type(generator)) # class : generator

# 方法1.调用生成器的方法__next__()得到元素
print(generator.__next__()) # 0
print(generator.__next__()) # 1 使用方法__next__()几次则返回几
# 方法2.调用next(g) 系统自带的函数得到元素
print(next(generator)) # 4 此函数如果和__next__()是一起使用的也会叠加次数
# 以此类推回得到9,16,...,19^2 列表的每个元素 直到超过元素个数会抛出异常 停止迭代错误
while 1:
    try:
       out = next(generator)
       print(out)
    except Exception as err :
       print(err)
       break

'''利用函数得到生成器'''
def  func():
     n = 0
     while 1:
         n+=1
         yield n
g = func() # 此函数为生成器对象
print(next(g))
print(next(g))

# 斐波那契数列 0 1 1 2 3 5 8 13 ...
def fib(length) :
    a , b = 0 ,1
    n = 0
    while n < length :
        yield b
        a , b = b , a+ b # 数列后一个元素总是前两个元素的和
        n+=1
    else:
        raise Exception('输入参数超过列表长度') # 如果生成器有异常返回-1作为提示
g = fib(8)
try:
   for i in range(9):
       print(next(g))
except Exception as err:
   print('错误类型为:',err)

def task1(n) :
    for i in range(n) :
        print('正在搬第{}块砖'.format(i))

#利用函数生成器实现多线程运行
def task2(n):
    for i in range(n):
        print('正在听第{}块歌'.format(i))
# 顺序执行
task1(8) 
task2(10)

def task3(n) :
    for i in range(n) :
        print('正在搬第{}块砖'.format(i))
        yield None

def task4(n):
    for i in range(n):
        print('正在听第{}块歌'.format(i))
        yield None
# 利用生成器实现多线程交替运行 
g1 = task3(8)
g2 = task4(10)
while True :
    try: 
       next(g1)
       next(g2)
    except :
        pass