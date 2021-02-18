# 正则表达式 : 正则表达式是对字符串操作的一种逻辑公式
# 目的 : 校验给定字符串是否满足正则表达式的过滤逻辑 称作匹配
# 还可通过正则表达式得到想要的部分
import re
'''
1.match 2.search 3.span 4.group 5.compile 6.findall
'''
msg = "chenbeicbccbbcb"
pattern = re.compile('chenbei') # 被匹配的东西
# print(pattern)
msg_match = pattern.match(msg)
print(msg_match) # match开头如果没有匹配则返回空 <re.Match object; span=(0, 7), match='chenbei'>
print(re.match('chenbei',msg)) # 或者一行代码也可以实现
result = re.search('cb',msg,2)
print(result) # 可以在所有字符串中找到是否存在满足的
t1 = result.span() # tuple
print(t1) # 返回位置
s1 = result.group() # string
print(s1) # 返回匹配的字符串

#############################################################################
msg ='32nd4fbdfdfg44f34nn5y4yn45ny'
result1 = re.search('[a-z][0-9][a-z]',msg) # 找到满足'[字母][数字][字母]'规定性质的第一组字符串
print(result1.group())
print(re.findall('[a-z][0-9][a-z]',msg)) # 找到所有匹配的

######################################################################
'''
msg = 'H2g423mfj4g3434g5'
1.正则的符号
'.' : 匹配除换行符\n 以外的所有字符 -->
'^' : 匹配字符串的开始 行首
'$' : 匹配字符串的末尾 如果末尾有换行符则匹配\n前边的那个字符 即行尾 
定义正则验证次数
'*' : 将前面的模式匹配0次或多次(贪婪模式 尽可能多匹配)
'+' : 将前面的模式匹配1次或多次(贪婪模式 尽可能多匹配)
'?' : 将前面的模式匹配0次或1次(贪婪模式)
'{m}' : 用于验证将前面的模式匹配m次
'{m,}': 用于验证将前面的模式匹配m次或者多次 >=m次
'{m,n}' : 将前面的模式匹配m次到n次(贪婪模式 最小匹配m次 最多匹配n次)
'{m,n}?' : 即上方{m,n}的非贪婪版本
'\\' : 特殊字符加上\ 就失去了原有含义 如\+ 只表示加号+本身
'[]' : 标示一组字符[0-9] u若首字符为^ 则表示补集 [^0-9]表示数字以外的字符
'|' : A|B 用于匹配A或B
'''
'''
\A : 表示从字符串的开始匹配
\Z : 表示从字符串的结束处匹配 存在换行只匹配到换行前的字符
\b : 匹配一个单词边界 指单词和空格间的位置 例如'py\b'可以匹配'python'的'py' 但不能匹配'openpyxl'的'py'
\B : 匹配非单词边界 'py\B'可以匹配'openpyxl'的'py' 但不能匹配'python'的'py'
\d : 匹配任意数字 等价于[0-9]
\D : 匹配任意非数字字符 等价于[^0-9]
\s : 匹配任意空白字符 等价于[\t\n\r\f]
\S : 匹配任意非空白字符 等价于[^\s]
\w : 匹配任意字母数字和下划线 等价于[a-zA-Z0-9_]
\W : 匹配任意非字母数字和下划线 等价于[^\w]
\\ : 匹配原义的反斜杠\
'''
# 用户名可以是字母或者数字 不能是数字开头 用户名长度必须6位以上
# [a-zA-Z] : 不能是数字开头 [0-9a-zA-Z]{5,} : 可以是数字或字母 5位以上
username1 = 'admin001'
username2 = '#@admin001'
username3 = 'admin001#&%'
username4 = '#&admin001%@#'
#%%
# match匹配
# 1. <re.Match object; span=(0, 8), match='admin001'>
print(re.match('[a-zA-Z][0-9a-zA-Z]{5,}',username1))
# 2.None : 问题在于从头匹配 不满足条件
print(re.match('[a-zA-Z][0-9a-zA-Z]{5,}',username2))
# 3.<re.Match object; span=(0, 8), match='admin001'> : 只考虑了头部匹配没有考虑尾部匹配
print(re.match('[a-zA-Z][0-9a-zA-Z]{5,}',username3))
#%%
# search查找  username1 = 'admin001' 输出全部相同 对于该字符串满足尾部和首部限制
# 4.1 <re.Match object; span=(0, 8), match='admin001'>
print(re.search('[a-zA-Z][0-9a-zA-Z]{5,}',username1))
# 4.2 <re.Match object; span=(0, 8), match='admin001'>
print(re.search('^[a-zA-Z][0-9a-zA-Z]{5,}',username1))
# 4.3 <re.Match object; span=(0, 8), match='admin001'>
print(re.search('[a-zA-Z][0-9a-zA-Z]{5,}$',username1))
# 4.4 <re.Match object; span=(0, 8), match='admin001'>
print(re.search('^[a-zA-Z][0-9a-zA-Z]{5,}$',username1))
#%%
# search查找 username2 = '#@admin001' 头部不匹配 尾部匹配
# 5.1 <re.Match object; span=(2, 10), match='admin001'> : 不加限制没有影响
print(re.search('[a-zA-Z][0-9a-zA-Z]{5,}',username2))
# 5.2 None : 加了首部匹配限制 所以返回None 首部不能是数字字母以外且首位不为0
print(re.search('^[a-zA-Z][0-9a-zA-Z]{5,}',username2))
# 5.3 <re.Match object; span=(2, 10), match='admin001'> : 加尾部匹配限制没有影响
print(re.search('[a-zA-Z][0-9a-zA-Z]{5,}$',username2))
# 5.4 None : 加了首部匹配限制 所以返回None 首部不能是数字字母以外且首位不为0
print(re.search('^[a-zA-Z][0-9a-zA-Z]{5,}$',username2))
#%%
# search查找 username3 = 'admin001#&%' 尾部不匹配 头部不匹配
# 6.1 <re.Match object; span=(0, 8), match='admin001'> : 不加限制没有影响
print(re.search('[a-zA-Z][0-9a-zA-Z]{5,}',username3))
# 6.2 <re.Match object; span=(0, 8), match='admin001'> : 加头部匹配限制没有影响
print(re.search('^[a-zA-Z][0-9a-zA-Z]{5,}',username3))
# 6.3 None : 加了尾部匹配限制 所以返回None 即尾部不能是数字字母以外的
print(re.search('[a-zA-Z][0-9a-zA-Z]{5,}$',username3))
# 6.4 None : 加了尾部匹配限制 所以返回None 即尾部不能是数字字母以外的
print(re.search('^[a-zA-Z][0-9a-zA-Z]{5,}$',username3))
#%%
# search查找 username4 = '#&admin001%@#'
# 7.1 <re.Match object; span=(2, 10), match='admin001'> : 不加限制没有影响
print(re.search('[a-zA-Z][0-9a-zA-Z]{5,}',username4))
# 7.2 None : 加了首部匹配限制 所以返回None 首部不能是数字字母以外且首位不为0
print(re.search('^[a-zA-Z][0-9a-zA-Z]{5,}',username4))
# 7.3 None : 加了尾部匹配限制 所以返回None 即尾部不能是数字字母以外的
print(re.search('[a-zA-Z][0-9a-zA-Z]{5,}$',username4))
# 7.4 None : 头部和尾部限制都不满足
print(re.search('^[a-zA-Z][0-9a-zA-Z]{5,}$',username4))