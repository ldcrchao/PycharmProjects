# file = open("C:\\Users\\chenbei\\Desktop\\钢\\data.txt") 'rt'模式
# read()函数
# print(file.read())  # 读取全部内容

# readable()函数
# print(file.readable())  # 是否可以读取,返回bool值

# readline()函数
# print(file.readline()) # 读取第一行
# while True :
#     line = file.readline() # 读取每一行
#     print(line)
#     if not line :
#         break  # 空行时退出

# readlines()函数 与read()区别在于不是按每行读取,且还读了换行符,返回的是列表,所有行
# lines = file.readlines()
# for line in lines:
#     print(line)

# f = open("C:\\Users\\chenbei\\Desktop\\钢\\dat.txt", 'w') # 写文件模式但是覆盖
#
# # write()函数
# f.write('chenbei\n')
# f.write('zuishuai\n')
#
# # writelines()函数
# f.writelines(['name\n', 'sex\n', 'birth\n'])  # 区别在于可以写入列表存放的多行
# f.close()  # 在open和close之间的会继续写入而不会覆盖,close后重新写入会清空之前全部内容并覆盖

# 写文件模式不覆盖,在原有基础上追加 'a'模式
# f = open("C:\\Users\\chenbei\\Desktop\\钢\\dat.txt", 'a')
# f.write('\ngirlfirend')
# f.close()

# with _ open 可以自动释放资源
# 文件复制
with open(r"C:\Users\chenbei\Desktop\钢\data.txt",'rt') as file :
    tem = file.read()
    with open(r"C:\Users\chenbei\Desktop\钢\dat.txt",'wt') as wf :
        wf.write(tem)

import os
abspath = os.path.abspath(r'C:\Users\chenbei\PycharmProjects\python学习工程文件夹\test.py')
print(abspath) # 绝对路径
path = os.path.realpath(r'C:\Users\chenbei\PycharmProjects\python学习工程文件夹\test.py')
print(path) # 实际路径
path1 = os.path.dirname(r'C:\Users\chenbei\PycharmProjects\python学习工程文件夹\test.py')
print(path1)# 当前文件所在的文件夹
path2 = os.path.join(path1,'test.py')
print(path2) # 使用join函数拼接

with open(r"C:\Users\chenbei\Desktop\钢\data.txt",'rt') as file :
    fname = file.name # 也可以返回该file所在的路径
    ff = fname.rfind('\\') # 返回字符串最后一次出现的位置，如果没有匹配项则返回 -1
    # ff = 26 最后一次出现在26位置,从位置0开始'C'
    ft = fname[ff+1:] # 切片,即得到目录的最后1级


# 相对路径表示相对于当前所在的文件test.py找到其它文件
# 绝对路径是找到c盘下的路径
# r = os.path.isabs('Car\Car.py') # 找到当前文件夹即test.py同级的文件或者文件夹,叫相对路径
# 特别的如果 test.py在某个文件夹中仍想找到Car.py文件 应当使用../
r = os.path.isabs(r'../Car/Car.py') # 找到上一级的同级文件夹Car的Car.py文件
# 如果还有文件夹可以使用 ../../Car/Car.py

r1 = os.path.dirname(os.path.abspath('__file__')) # 返回当前文件所在目录的上一级目录
r2 = os.getcwd() # r1 = r2
r3 = os.path.abspath('test.py') # 根据相对路径得到绝对路径

r4 = os.path.split(os.getcwd()) # 会分割给定的路径最后一级和以上的目录路径
# 'C:\\Users\\chenbei\\PycharmProjects', 'python学习工程文件夹'

r5 = os.path.splitext(r'C:\Users\chenbei\PycharmProjects\python学习工程文件夹\测试文件夹\test.py') # 分割拓展名
# 'C:\\Users\\chenbei\\PycharmProjects\\python学习工程文件夹\\测试文件夹\\test', '.py'

# 拼接路径
r6 = os.path.join(os.getcwd(),'常用规则','索引规则.py')

# 列出工程文件夹下的当前所有文件夹和文件
r7 = os.listdir(os.getcwd())

# 判断是否存在该目录
r8 = os.path.exists(os.path.join(os.getcwd(),'创建文件夹'))

# 创建文件夹
r9 = os.mkdir(os.path.join(os.getcwd(),'创建文件夹') )

# 只能删除空的文件夹
r10 = os.rmdir(os.path.join(os.getcwd(),'创建文件夹') )

# 只能删除空的文件夹
r11 = os.mkdir(os.path.join(os.getcwd(),'创建文件夹') )
r12 = os.removedirs(os.path.join(os.getcwd(),'创建文件夹') )

# 删除文件的函数
# os.remove()
# 遍历删除某个文件夹下的文件的伪代码
# path = ''
# for file in os.listdir(path) :
#     newfile = os.path.join(path,file) # 可以得到目录下所有的文件的绝对路径
#     os.remove(newfile) # 遍历删除某个文件夹下的文件
# else :
#     os.rmdir(path) # 如果该文件夹下没有文件,则直接删除该空文件夹


# 复制文件夹
def copyfile(src,target) :
    if os.path.isdir(src) and os.path.isfile(target) : # 判断是否为文件夹
        filelist = os.listdir(src) # [a.taxt,b.txt,...]
        for file in filelist : # a.txt
            path = os.path.join(src,file) # 源路径拼接文件名
            with open(path,'rt') as f1 : # 打开这个文件
                text = f1.read()
                with open (target,'w') as f2:
                    f2.write(text)
        else :
             print('复制完毕')
    else :
        if not os.path.isdir(src) :
           print('您选择的源文件路径不是文件夹')
        else:
            print('您选择的源文件路径是文件夹')
        if not os.path.isfile(target) :
            print('您的源文件和目标文件路径均错误')
        else:
            print('您选择的目标文件路径是文件')
copyfile(r'C:\Users\chenbei\PycharmProjects\python学习工程文件夹\robot.csv',\
         r'C:\Users\chenbei\PycharmProjects\python学习工程文件夹\robot.csv')


