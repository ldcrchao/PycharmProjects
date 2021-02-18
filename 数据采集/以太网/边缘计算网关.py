#%%
import socket
import struct
import time
import datetime
import pandas as pd
import  numpy  as np
"""
数据采集卡地址'192.168.1.198'
端口 ：1600
"""
def ADset(tcpClient):
    '''设置同步采集参数指令结构表:一共36位的tcp字段'''
    package_code = [0x06,0x5a]  # 校验码 设置同步采集指令码的字段取值：0x5a05
    nopforward = [0x00,0x00] #备用
    ad_frequency = [0xc0,0xd4,0x01,0x00] #ad采样频率 0x7d00=32000 007d 0000
    # 0x00013880 = 80000 , 0001应该对应0000的位置，且高低交换为0100 ，最终8038 0100
    # 0x0001d4c0 = 120000 , 最终c0d4 0100
    ad_page = [0x00,0x00,0x00,0x00]   #程控放大倍数
    nopmiddle_1 = [0x00,0x00,0x00,0x00]
    nopmiddle_2 = [0x00,0x00,0x00,0x00]
    nopmiddle_3 = [0x00,0x00,0x00,0x00]
    nopmiddle_4 = [0x00,0x00,0x00,0x00]
    nopmiddle_5 = [0x00]
    ad_start_channel = [0x00]
    ad_end_channel = [0x07]
    mux_type = [0x00] # 0-采用单端输入采集方式 ，1-双端输入采集方式
    nopbackward = [0x00,0x00,0x00,0x00]
    setting = package_code + nopforward + ad_frequency + ad_page + nopmiddle_1 + nopmiddle_2 + nopmiddle_3 + nopmiddle_4 + nopmiddle_5 + ad_start_channel + ad_end_channel + mux_type + nopbackward

    '''打包消息'''
    setting_message = struct.pack("%dB" % (len(setting)), *setting) # %d的位置对应后边列表的长度 ，结果是36B，表示可以存贮一个长度为36的无符号字符串
    #import math ; print("%.3fB" % math.pi) 表示一种格式化的写法，即以指定格式格式化，% 用于连接被格式化的对象 , out = [3.142B]

    '''发送消息'''
    tcpClient.send(setting_message)

    '''接收返回结果'''
    setting_response = tcpClient.recv(64)

    '''解析返回结果'''
    setting_result = struct.unpack("%dB" % (len(setting_response)), setting_response) # 校验码 ，也就是发送时首个字段package_code
    #print([hex(setting_result[0]), hex(setting_result[1])]) #按字节返回十进制 ，hex转换成16进制
    if [setting_result[0], setting_result[1]] == [setting_message[0], setting_message[1]]:
        print('返回指令包与发送指令包相等，设置采集成功')
        return 1
    else:
        print('"设置启动指令错误"')
        return 0

def ADstop(tcpClient):
    '''停止采集指令结构表'''
    package_code = [0xa2, 0x5a]  # 指令符
    nop = [0x00,0x00,0x00,0x00,0x00,0x00]  # 占位符
    read = package_code + nop
    ##########################
    # 打包消息
    read_message = struct.pack("%dB" % (len(read)), *read)
    # 发送消息
    tcpClient.send(read_message)
    # 接收返回结果
    read_response = tcpClient.recv(1900)
    #print(read_response)  #b'\xa2Z\xce\xdc\x00\x00\x00\x00' 除了头2个字节不变其它都会随机变化
    # 解析返回结果
    read_result = struct.unpack("%dB" % (len(read_response)), read_response)
    return read_result   #(162, 90, 206, 220, 0, 0, 0, 0) = (a2, 5a, ce, dc, 00 , 00 ,00 ,00 ) 除了头2个字节不变其它都会随机变化

def ADstart(tcpClient):
    '''启动采集指令结构表'''
    package_code = [0xa1, 0x5a]  # 启动采集指令码
    nop = [0x00, 0x00]  # 备用占位符
    ad_type = [0x00, 0x00, 0x00, 0x00]  #A/D 采集类型

    start = package_code + nop + ad_type
    ##########################
    # 打包消息
    start_message = struct.pack("%dB" % (len(start)), *start)
    # 发送消息
    tcpClient.send(start_message)
    # 接收返回结果
    start_response = tcpClient.recv(64)
    # 解析返回结果
    start_result = struct.unpack("%dB" % (len(start_response)), start_response)
    #print([hex(start_result[0]), hex(start_result[1])])
    if [start_result[0], start_result[1]] == [start_message[0], start_message[1]]:
        print('返回指令包与发送指令包相等，启动采集成功')
        return 1
    else:
        print('启动采集命令错误')
        return 0

def ADDataRead(tcpClient):
    '''读采集数据指令结构表'''
    package_code = [0xa4, 0x5a]  # 停止采集指令码
    nop = [0x00, 0x00]  # 备用
    ulLength = [0xc0, 0xd4, 0x01, 0x00]  # 指定读取采集数据的个数  0—2^32 ,但是每次最多发送720个数据 02d0 = 720 ，这里指定接受 120000个数

    read = package_code + nop + ulLength
    ##########################
    # 打包消息
    read_message = struct.pack("%dB" % (len(read)), *read)
    # 发送消息
    tcpClient.send(read_message)
    # 接收返回结果
    read_response = tcpClient.recv(1448)
    # 解析返回结果
    read_result = struct.unpack("%dB" % (len(read_response)), read_response)
    # 删除头8个字节
    #read_result = read_result[8:-1] 这样会丢掉最后1个数
    #read_result = read_result[8:]
    k1 = np.arange(0, len(read_result), 1448)  # 0 ,1448,2896...28960
    k3 = []
    for i in read_result:
        k3.append(i)  # 先转换成列表就可以删除每1448个字节的头8个
    for i in k1:
        del k3[i:i + 8]  # 应当删除8*(len(read_result)/1448)  这么多个头8字节
    return k3

def list_to_hex_to_str(List):
    '''将列表类型转化为连续的字符串'''
    Hex = [hex(i) for i in List]
    string = [str(j) for j in Hex]
    string = ''.join(string)
    string = string.replace('0x', '')
    return string

def hex_to_signed_10(data):
    '''16进制 转换为有符号十进制'''
    width = 32  # 16进制数所占位数
    data = 'FFFF' + data
    dec_data = int(data, 16)
    if dec_data > 2 ** (width - 1) - 1:
        dec_data = 2 ** width - dec_data
        dec_data = 0 - dec_data
    dec_data = 5*( -1 * dec_data / 3280.7 )/8#参考labview的缩放倍数
    return dec_data

def hex_to_signed_10_much(string):
    '''string 应当是长度360的字符串'''
    String = []
    k = np.arange(0,len(string),4) #0,4,8,...,352,356的等差数列
    k = k.astype(int)
    for i in k:
        temp = hex_to_signed_10(string[i:i+4])
        String.append(temp)    #返回的是列表类型
    return String

def csv_tuple_str(temp_data):
    '''数据1440个字节 -> 每个通道180字节 -> 每个通道360个16进制数 -> 4个16进制也就是2个字节1个数,每个通道90个有符号十进制数'''
    '''所有通道 90 × 8 = 720 个数'''
    dict_tmp = {}
    Dict_tmp ={}
    dec = {}
    time.sleep(0.1)  # 启动采集指令之后加一个100ms延时
    dt = datetime.datetime.now()
    dict_tmp['time'] = dt
    Dict_tmp['time'] = dt
    dec['time'] = dt
    i = 0
    channel_0 = []  # 列表类型
    channel_1 = []
    channel_2 = []
    channel_3 = []
    channel_4 = []
    channel_5 = []
    channel_6 = []
    channel_7 = []
    h = np.arange(0, len(temp_data), 8)  # 分成8个通道
    h = h.astype(int)  # 后续切片必须是整数
    for i in h:
        channel_0.append(temp_data[i + 0])
        channel_1.append(temp_data[i + 1])
        channel_2.append(temp_data[i + 2])
        channel_3.append(temp_data[i + 3])
        channel_4.append(temp_data[i + 4])
        channel_5.append(temp_data[i + 5])
        channel_6.append(temp_data[i + 6])
        channel_7.append(temp_data[i + 7])
    # 元组类型
    dict_tmp['channel_0'] = channel_0
    dict_tmp['channel_1'] = channel_1
    dict_tmp['channel_2'] = channel_2
    dict_tmp['channel_3'] = channel_3
    dict_tmp['channel_4'] = channel_4
    dict_tmp['channel_5'] = channel_5
    dict_tmp['channel_6'] = channel_6
    dict_tmp['channel_7'] = channel_7
    # 16进制的字符串类型 ，长度360
    Channel_0 = list_to_hex_to_str(channel_0)
    Channel_1 = list_to_hex_to_str(channel_1)
    Channel_2 = list_to_hex_to_str(channel_2)
    Channel_3 = list_to_hex_to_str(channel_3)
    Channel_4 = list_to_hex_to_str(channel_4)
    Channel_5 = list_to_hex_to_str(channel_5)
    Channel_6 = list_to_hex_to_str(channel_6)
    Channel_7 = list_to_hex_to_str(channel_7)
    # 字典类型
    Dict_tmp['channel_0'] = Channel_0
    Dict_tmp['channel_1'] = Channel_1
    Dict_tmp['channel_2'] = Channel_2
    Dict_tmp['channel_3'] = Channel_3
    Dict_tmp['channel_4'] = Channel_4
    Dict_tmp['channel_5'] = Channel_5
    Dict_tmp['channel_6'] = Channel_6
    Dict_tmp['channel_7'] = Channel_7
    # 有符号十进制类型
    Channel_0_10 = hex_to_signed_10_much(Channel_0)
    Channel_1_10 = hex_to_signed_10_much(Channel_1)
    Channel_2_10 = hex_to_signed_10_much(Channel_2)
    Channel_3_10 = hex_to_signed_10_much(Channel_3)
    Channel_4_10 = hex_to_signed_10_much(Channel_4)
    Channel_5_10 = hex_to_signed_10_much(Channel_5)
    Channel_6_10 = hex_to_signed_10_much(Channel_6)
    Channel_7_10 = hex_to_signed_10_much(Channel_7)
    dec['channel_0'] = Channel_0_10
    dec['channel_1'] = Channel_1_10
    dec['channel_2'] = Channel_2_10
    dec['channel_3'] = Channel_3_10
    dec['channel_4'] = Channel_4_10
    dec['channel_5'] = Channel_5_10
    dec['channel_6'] = Channel_6_10
    dec['channel_7'] = Channel_7_10
    print(dict_tmp)
    # print(Dict_tmp)
    #print(dec)
    return dt,dec,dict_tmp

if __name__ == '__main__':
    '''建立tcp连接'''
    tcpClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 客户端
    tcpClient.connect(('192.168.1.198', 1600))
    '''停止AD采样'''
    #ADstop(tcpClient)
    '''设置采集命令'''
    Settings = ADset(tcpClient)  # 成功时返回值 = 1
    #if not Settings:
    #   print("设置启动指令错误")
    '''启动采集命令'''
    Start = ADstart(tcpClient)
    #if not Start:
    #    print("启动采集命令错误")
    sample_time_start = datetime.datetime.now()
    print("数据采集开始", sample_time_start)
    sample_time_end = sample_time_start + datetime.timedelta(seconds=1)
    print("数据采集计划结束", sample_time_end)
    DATA_tuple = [] #16进制元组
    #DATA_str = []   #16进制字符串
    DATA_signed10 = [] #有符号十进制数
    while True:
        '''数据接收'''
        temp_data  = ADDataRead(tcpClient) #1440个字节
        '''数据处理'''
        dt ,dict_10,dict_tmp = csv_tuple_str(temp_data)
        DATA_tuple.append(dict_tmp)
        #DATA_str.append(Dict_tmp)
        DATA_signed10.append(dict_10)
        time_now = datetime.datetime.now()
        if dt > sample_time_end:
            print("数据存储结束", time_now)
            break
        df_tuple = pd.DataFrame(DATA_tuple)
        #df_str = pd.DataFrame(DATA_str)
        df_signed_10 = pd.DataFrame(DATA_signed10)
        df_tuple.to_csv("C:/Users\chenbei\Desktop\钢\data_tuple.csv",index= False ) #不保存标签
        #df_str.to_csv("C:/Users\chenbei\Desktop\钢\data_str.csv", index=False)
        df_signed_10.to_csv("C:/Users\chenbei\Desktop\钢\data_signed_10.csv", index=False)
    tcpClient.close()

#%%



