"""
client.py
encode()/decode()
"""
from socket import *
import struct
import time
import datetime
import pandas as pd
import pickle

"""
1 车身（焊接）车间 升降机IP地址 "172.19.64.100"  端口：1600
2 涂装车间水泵IP地址："172.20.1.181" 端口：1600  超滤循环泵1
          风机IP地址："172.20.1.184" 端口：1600  CC内排风机

"""


#同步采集仪A/D设置函数
def ADSyncParaWrite(udpClient):
    package_code = [0x05, 0x5a]  # 字段取值：0x5a05
    nop = [0x00, 0x00]  # 占位符
    ad_frequency = [0xe8, 0x03, 0x00, 0x00]  # 1000Hz采样频率
    # ad_range = [0x01, 0x00, 0x00, 0x00]  # 正负10v
    ad_range = [0x00, 0x00, 0x00, 0x00]  # 正负5v

    ain_select = [0x00, 0x00, 0x00, 0x00]  # 模拟信号输入 0 DC直流电 1 AC交流电
    ch_enabled0 = [0xff,0xff]  # 全部通道一起开启  升降机使用的是8通道的采集仪

    master_flag = [0x00, 0x00]  # 主控板
    setting = package_code + nop + ad_frequency + ad_range + ain_select + ch_enabled0 + master_flag
    ##########################
    # 打包消息
    setting_message = struct.pack("%dB" % (len(setting)), *setting)
    # print("setting_message:",hex(setting_message[0]),hex(setting_message[1]))
    # 发送消息
    udpClient.send(setting_message)
    # 接收返回结果
    setting_response = udpClient.recv(64)
    # 解析返回结果
    setting_result = struct.unpack("%dB" % (len(setting_response)), setting_response)
    print([hex(setting_result[0]), hex(setting_result[1])])
    if [setting_result[0], setting_result[1]] == [setting_message[0], setting_message[1]]:
        # print('返回指令包与发送指令包相等，设置采集成功')
        return 1
    else:
        return 0

def ADStart(udpClient):
    # 启动数据采集 7 启动采集指令结构表
    package_code = [0xa1, 0x5a]  # 指令符
    nop = [0x00, 0x00]  # 占位符
    ad_type = [0x00, 0x00, 0x00, 0x00]  # DC 交流电

    start = package_code + nop + ad_type
    ##########################
    # 打包消息
    start_message = struct.pack("%dB" % (len(start)), *start)
    # 发送消息
    udpClient.send(start_message)
    # 接收返回结果
    start_response = udpClient.recv(64)
    # 解析返回结果
    start_result = struct.unpack("%dB" % (len(start_response)), start_response)
    print([hex(start_result[0]), hex(start_result[1])])
    if [start_result[0], start_result[1]] == [start_message[0], start_message[1]]:
        # print('返回指令包与发送指令包相等，启动采集成功')
        return 1
    else:
        return 0

def ADDataRead(udpClient):
    package_code = [0xa4, 0x5a]  # 指令符
    nop = [0x00, 0x00]  # 占位符
    ulLength = [0xd0, 0x02, 0x00, 0x00]  # DC 交流电

    read = package_code + nop + ulLength
    ##########################
    # 打包消息
    read_message = struct.pack("%dB" % (len(read)), *read)
    # 发送消息
    udpClient.send(read_message)
    # 接收返回结果
    read_response = udpClient.recv(1900)
    # 解析返回结果
    read_result = struct.unpack("%dB" % (len(read_response)), read_response)
    return read_result          #read_result

    # print([hex(read_result[0]), hex(read_result[1])])

def ADStop(udpClient):
    package_code = [0xa2, 0x5a]  # 指令符
    nop = [0x00, 0x00]  # 占位符
    ulLength = [0xd0, 0x02, 0x00, 0x00]  # DC 交流电

    read = package_code + nop + ulLength
    ##########################
    # 打包消息
    read_message = struct.pack("%dB" % (len(read)), *read)
    # 发送消息
    udpClient.send(read_message)
    # 接收返回结果
    read_response = udpClient.recv(1900)
    # print(read_response)


    # 解析返回结果
    read_result = struct.unpack("%dB" % (len(read_response)), read_response)
    return read_result

    # print([hex(read_result[0]), hex(read_result[1])])





if __name__=="__main__":
    #建立UDP连接
    udpClient = socket(AF_INET, SOCK_DGRAM)

    # 开始连接服务端IP和PORT,建立双向链接
    ######升降机UDP设置#############
    udpClient.connect(('172.20.1.184', 1600))  # 通过服务端IP和PORT进行连接

    #先停止AD采样
    ADStop(udpClient)            #在设置同步采集指令前，增加停止采集指令

    #设置同步采集指令
    ParaWrite = ADSyncParaWrite(udpClient)
    if not ParaWrite:
        print("设置指令错误")

    Start = ADStart(udpClient)
    if not Start:
        print("采集错误")
    sample_time_start = datetime.datetime.now()  # 设置振动开始采集的时间
    print("数据采集开始",sample_time_start)
    sample_time_end = sample_time_start + datetime.timedelta(minutes=10)
    print("数据采集计划结束",sample_time_end)
    v_lst = []
    while True:
        dict_tmp = {}
        time.sleep(0.1)  # 启动采集指令之后加一个100ms延时
        # tstmp = time.time()
        dt = datetime.datetime.now()
        dict_tmp['time'] = dt
        dict_tmp['v_value'] = ADDataRead(udpClient)
        # print(dict_tmp)
        # v_lst.append(dict_tmp)
        with open('D:\风机数据保存\风机数据3.txt', 'a') as f:
            f.write('\n')  # 换行
            f.write(str(dict_tmp))
            f.close()
        time_now = datetime.datetime.now()
        if dt > sample_time_end:  # 1分钟后结束程序
            print("数据存储结束",time_now)
            break
    # df = pd.DataFrame(v_lst)
    # df.to_csv("D:\水泵数据保存\水泵振动数据_1min.csv")









