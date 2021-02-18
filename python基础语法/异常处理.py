# try : 正常执行时执行的代码块 except: 如果有异常执行的代码 finally : 无论是否异常都会执行的代码
# 可以保证异常时后续的代码仍能运行不会中断
# 系统抛出异常
def func(length):
    try :
        if length >= 18 :
            print('顺哥的鸡鸡还行不太短')
        else:
            print(length / 0)
    # func(18) 如果输入的参数不对会抛出类型错误异常,然后会找到对应的异常类型执行相应代码块
    except TypeError : 
        print('输入必须为int或float型')
    # func(2) 这里应当是ZeroDivisionError错误,但是没给的话会找到Exception,它是所有错误类型的父类
    # except ZeroDivisionError :
    #     print('分母不能是0')
    except  Exception as err: # 放在最后
        print('错误类型为:',err)
    finally:
        print("无论异不异常顺哥都没北哥的大")
func(2)

import numpy as np
def beauty(yours,others):
    if isinstance(yours,str) and isinstance(others,str) :
       yours = bool(yours)
       others = bool(others)
    # 量化美貌
    def decode_beauty():
        if yours :
            yoursscore = 100
            print('还用说,如果是你的话当然是{}分'.format(yoursscore))
        if others :
            othersscore =  np.random.randint(0,60)
            print('其他人？就这,随便打个及格分以下把就,%d分不能再多了' % othersscore )
        return yoursscore , othersscore
    try :
        yours_score , others_score = decode_beauty()
        subtract  = yours_score - others_score
        print(f'很明显,你比别人美了这么多{subtract}倍')
    except  Exception as err :
        print(err)
    finally:
        print('无论和谁比,if you ,你都是最美的！')
beauty('You','AnyOtherName')

# 手动抛出异常
# 抛出异常
def register():
    username = input('请输入用户名:')
    if len(username) < 6 :
        raise Exception('用户名必须为6位') # 能够抛出异常就可以使用try-except接收
    else:
        print(username)
def isSuccess():
    try:
       register() # 使用try-except接收
    except Exception as err:
        print(err)
        print('注册失败')
        return -1
    else: # try - except - else 结构也可以
        print('注册成功')
        return 1
isSuccess()
