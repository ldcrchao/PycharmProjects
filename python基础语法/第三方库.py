import datetime
import  time
import  random
import requests
import  hashlib
response = requests.get("http://www.12306.cn/")
#print(response.text)
# 加密算法 md5 sha1 sha256 不可逆 base64 可逆
msg = 'chenbei'
msg_md5 = hashlib.md5(msg.encode('utf-8'))
print(msg_md5.hexdigest()) # 加密数据的16进制

t1 = time.time()
print(t1 ) # 时间戳(float) : 1612690096.4199407

t2 = time.ctime(t1)
print(t2) # 本地时间(string) : Sun Feb  7 17:29:48 2021

t3 = time.localtime(t1)
print(t3) # 元组形式(tuple)

t4 = time.mktime(t3)
print(t4) # 元组转成时间戳 : 精度降低

# %Y %m %d %H %M %s
t5 = time.strftime('%Y-%m-%d %H:%M:%S')
print(t5) # 打印出特定格式(string)


t6 = time.strptime('2019/06/20','%Y/%m/%d')
print(t6.tm_year)# 字符串转为元组

t7 = datetime.time
print(t7) # <class 'datetime.time'>
t8 = t7.hour
# print(t8) # <attribute 'hour' of 'datetime.time' objects>
print(str(t8))# <attribute 'hour' of 'datetime.time' objects>

t9 = datetime.date(2019,6,20)# date格式
t10 = t9.today()
print(t10)# date格式 2021-02-07

# 时间差值 用于会话机制 一定时间后清除缓存
t11 = datetime.timedelta(hours=2)
t12 = datetime.datetime.now()
print(t12)
print(t11+t12)

print(random.random() )# 0~1 随即小数
print(random.randrange(1,20,2)) # 指定范围和步长随机整数
print(random.randint(1,20)) # 指定范围和步长随机整数
print(random.choice(['12','23','34'])) # 随机抽取

List =[1,2,3,4,5,6]
random.shuffle(List)
print(List) # 打乱顺序 洗牌 保证随机性