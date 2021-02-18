# 格式 
# 1.[表达式 for 变量 in 原列表 if 条件]
# 2.[结果A if 条件 else 结果B for 变量 in 原列表]
# 集合推导式 元素不重复，其它类似于列表推导式{}
# 字典推导式 ： {key ,values for k ,v in dict.iitems()}

names = ['chenbei','zhangliping','chenqifu','cb']
newnames = [x.capitalize() for x in names if len(x)<=8]

# 使用匿名函数过滤得到符合条件的列表,这个列表作为旧列表被改为大写
new1 = [i.capitalize() for i in list(filter(lambda x : x if len(x) <= 8 else None , names) )] # 回顾匿名函数和滤除函数

# 成对返回,x为偶数在(0,5)内,y为奇数在(0,10)内
new2 = [(x,y) for x in range(5) if x % 2 == 0 for y in range(10) if y % 2 == 1]

list1 = [[2,3,5],[3,3,6],[5,7,8],[3,5,8]] # 要求返回 [5 6 8 8]
new3 = [x[-1] for x in list1 ]
#%%
dict1 = {'name':'a','money':1000}
dict2 = {'name':'b','money':1500}
dict3 = {'name':'c','money':900}
dict4 = {'name':'d','money':800}
list2 = [dict1,dict2,dict3,dict4]
hhh = dict1['name']

# [x for x in list2] # 这里的x是每一个字典
# [x['money']+200 for x in list2]# 把x整体用表达式替换+200或者+400
# [x['money']+200 if x['money'] >=1000 else x['money']+400 for x in list2]
# 把条件给出来,利用三目运算符这一步实际上已经得到了一个满足条件的列表
new4 = [x['money']+200 if x['money'] >=1000 else x['money']+400 for x in list2]
#list2的每一项利用index可以得到其元素位置
new5 = [ {'name':x['name'],'money' : new4[list2.index(x)] } for x in list2] 
# 前边的表达式在关键字money后边 利用了new4返回的值

# 类似的字典推导式子 互换关键字和值
dict = {'a':'A','b':'B','c':'C','d':'D'}
newdict = { value : key      for key, value in dict.items()}


