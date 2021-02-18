#%%
import socket
import sys
import numpy as np
def main():
    s = socket.socket(socket.AF_INET , socket.SOCK_STREAM ) #客户端
    #host = '192.168.1.198' #服务器/以太网助手/数据采集卡的IP地址
    host = '192.168.16.103'
    port = 1600
    s.connect((host,port))
    setcommand = '065A 0000 007D 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0000 0700 0000 0000'
    responsecommand = 'A15A 0000 0000 0000'
    startcommand = 'A45A 0000 007D 0000'
    s.send(setcommand.encode())
    s.send(responsecommand.encode())
    s.send(startcommand.encode())
    s.sendto(b'123', (host, port))
    while 1:
        '''
        s.send(setcommand.encode())
        s.send(responsecommand.encode())
        s.send(startcommand.encode())
        '''
        A = s.recv(1448)
        B = A.decode()
        C = np.array(B)
        print(C)
        if not A:
           break
    s.close()
'''运行自身程序时，if __name__ '__main__'以下的程序将被执行 '''
'''如果外部调用该程序，if __name__ '__main__'以下的程序不被执行'''
if __name__ == '__main__':
    main()

#%% 服务端代码 服务端先运行
'''cmd输入命令：netstat -an|find /i "50000" '''
from socket import *
def server():
    listensocket = socket(AF_INET,SOCK_STREAM )#分别表示网络层和传输层的TCP协议，服务端
    IP = '192.168.16.103' #服务端提供的ip地址和端口号，用于客户端去连接
    PORT = 40000
    BUFLEN = 512
    listensocket.bind((IP,PORT)) #现在绑定好可以等待连接了
    listensocket.listen(5) #很多客户端在等待，最多有5个在等待
    print(f'服务器启动成功，正在{PORT}端口等待客户端连接...')
    '''第一次握手，服务端告诉客户端可以进行发送数据，等待回应'''
    datasocket , addr = listensocket.accept() #持续等待，返回的是数据和客户端地址
    print('一个客户端正在请求，地址为：',addr) #直到此语句运行，才会执行下方语句
    while 1:
          '''开始读取对方信息，最多读取BUFLEN个字节'''
          recv = datasocket.recv(BUFLEN)
          if not recv:
                break #如果收到空bytes说明对方关闭了连接，退出循环
          '''接收的数据是字节数据bytes类型，需要先解析为字符串'''
          info = recv.decode()
          print(f'收到对方信息为：{info}')
          '''第三次握手，服务端通知客户端已经收到成功数据，告诉对方已经收到信息'''
          datasocket.send(f'服务端已经收到信息{info}'.encode())
    datasocket.close() #关闭客户端请求连接
    listensocket.close() #关闭服务器监听
if __name__ == '__server__':
    server()
#%%客户端代码 另一台电脑
from socket import *
def client():
    IP = '192.168.16.103'
    ServerPort = 40000
    BUFLEN = 1024
    dataSocket = socket(socket.AF_INET , socket.SOCK_STREAM ) #客户端
    '''第二次握手，客户端连接服务器，通知服务器开始发送数据，提醒准备接受'''
    dataSocket.connect((IP,ServerPort)) #持续连接服务端
    while 1:
          '''等待用户在终端输入数据'''
          tosend = input('>>>')
          if tosend == 'exit':
                break
          '''发送数据'''
          dataSocket.send(tosend.encode() )
          '''第三次握手：等待服务端通知成功接收数据'''
          recved = dataSocket.recv(BUFLEN)
          '''如果返回空bytes，说明服务端已经关闭连接'''
          if not recved:
               break
          '''打印消息，客户端确认服务端能够成功收到数据'''
          print(recved.decode())
    dataSocket.close() #关闭客户端的主动请求
if __name__ == '__client__':
    client()


'''
应用消息格式：消息头（消息长度、类型、状态），消息体（具体的传送数据）
指定消息的边界：①用特殊字符作为消息的结尾符号，如'FFFFFF' ②在消息开头某个位置指定消息的长度
'''
'''①TCP长连接通讯②中途断开必须重连③每个消息都是UTF8编码的字符串'''
'''控制命令：①pause：暂停数据采集②resume：恢复数据采集'''
'''数据上报：汇报采集的数据，对方收到数据必须回复一个响应消息'''
'''
定义消息头：只包含一个信息，即消息体的长度
消息头用十进制的字符串表示一个整数的长度，即'200'表示3个字节,'20'2个字节
这是因为字节流中，字符'2'在ASCII码对应十六进制(HEX)的'32'，从而真正传递的是'323030'，所以占据3个字节
如果只用1个字节表示就是，即'200'直接变成十六进制的'C8'也是够用的，这样只占据1个字节，但是对人阅读理解不好，'200'是更好理解的
消息体使用json格式的字符串表示数据信息
'''
'''现在定义RUS为服务端，AT客户端，数据消息几类格式如下'''
'''
数据上报 RUS -> AT  服务端 -> 客户端
{
  "type" : "report"
  "info" : {
     "CPU Usage" : "30%"，
     "Mem Usage" : "54%"
  }
}
数据上报响应 AT -> RUS  客户端 -> 服务端
{
   "type" : "report-ack"
}
暂停数据上报命令 AT -> RUS  客户端 -> 服务端
{
  "type" : "pause"，
  "duration" : 200     #duration为暂停上报的时间，单位秒/s
}
恢复数据上报命令 AT -> RUS   客户端 -> 服务端
{
   "type" : "resume"
}
命令响应 RUS -> AT 服务端 -> 客户端
{
     "type" : "cmd-ack"，
     "code" : 200 ，  #code为结果处理代码，200表示成功
     "info" : "处理成功"
}
'''
'''
消息头格式
开头2个字节表示消息的长度，如00C8
第3个字节表示消息的数据类型：如已经定义命令代号为：0：pause暂停命令，1：resume恢复命令，2：report-ack命令响应命令，3：数据上报report命令，4：数据上报响应cmd-ack命令
注；如果消息不够长，2字节(16bit)描述消息长度过于浪费的话可以腾出3bit用于指定命令代号，即只前13bit用于传递消息长度。
如'200'表示为0000|0000 1100|1000 = 00C8高低位互换为C800 = 1100|1000 0000 0000 过于浪费， 可以调整为1100|1000 0000|0100 = C803(数据上报命令)，2字节就可以发送消息长度+命令代号，或者说消息长度用2个字节浪费，一个字节即可
消息体格式：
可以指定0："CPU Usage"，1："Mem Usage"，2："duration"，3："code" 
那么"CPU Usage" : "30%" 命令可以表示为00011E，头2位'00'表示"CPU Usage"命令代号，中间'01'表示命令格式是值的长度，也就是'30%'占据的长度，后面'1E'是真实数据
同样"Mem Usage" : "54%" 可以表示为010136，合起来就是00011E010136 ，需要6个字节表示2个信息
假如每次规定发送2个信息，则综合消息格式为060300011E010136 ，表示消息长度为6字节，命令类型为数据上报命令
'''
#%%
import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
num_points = 50
points_coordinate = np.random.rand(num_points, 2) # generate coordinate of points
distance_matrix = spatial.distance.cdist(points_coordinate, points_coordinate, metric='euclidean')
def cal_total_distance(routine):
    '''The objective function. input routine, return total distance. cal_total_distance(np.arange(num_points)) '''
    num_points, = routine.shape
    return sum([distance_matrix[routine[i % num_points], routine[(i + 1) % num_points]] for i in range(num_points)])
from sko.GA import GA_TSP
ga_tsp = GA_TSP(func=cal_total_distance, n_dim=num_points, size_pop=50, max_iter=500, prob_mut=1)
best_points, best_distance = ga_tsp.run()
fig, ax = plt.subplots(1, 2)
best_points_ = np.concatenate([best_points, [best_points[0]]])
best_points_coordinate = points_coordinate[best_points_, :]
ax[0].plot(best_points_coordinate[:, 0], best_points_coordinate[:, 1], 'o-r')
ax[1].plot(ga_tsp.generation_best_Y)
plt.show()


