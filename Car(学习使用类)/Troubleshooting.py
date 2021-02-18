# -*- coding: UTF-8 -*-
'''
8组故障，每组故障采集100组数据 ，放在1个excel/csv ，存放形式为列，每列是一组故障数据，长度控制在2000~3000且要一致
每组故障的每一组数据(列)都会返回1组 1×8的 数组，即每组故障都会得到100×8的特征数据
那么8组就可以得到800×8的特征数据，然后打乱选择测试/训练集的比例 转换为dataframe格式，并打上对应的标签即可进行训练
设 : 实际为真预测为真数量TT ，实际为真预测为假数量TF ； 实际为假预测为真FT ， 实际为假预测为假FF
准确率 Accuracy = (TT + FF) / (TT + TF + FT +FF) ，只要预测正确就行
精准率 Precise = TT / (TT + FT) 预测为真且确实为真的比例 ，在预测为真中 实际为真为假作为分母 : 衡量预测为真 ，且确实是真的比例
召回率 Recall = TT / (TT + TF) 实际为真且预测为真，在实际为真中 预测为真为假作为分母  : 衡量在为真的对象中，被发现为真的比例
'''
import socket
import struct
import time
import datetime
import pandas as pd
import random
import seaborn as sns
from scipy import signal
from sklearn.preprocessing import LabelEncoder
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy import interpolate
import pywt
import pywt.data
from pycaret.classification import *
from matplotlib.pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文的命令
mpl.rcParams['axes.unicode_minus'] = False  # 显示负号的命令


class tcp():
    def __init__(self, tcpClient):
        self.tcpClient = tcpClient

    def ADset(self):
        '''设置同步采集参数指令结构表:一共36位的tcp字段'''
        package_code = [0x06, 0x5a]  # 校验码 设置同步采集指令码的字段取值：0x5a05
        nopforward = [0x00, 0x00]  # 备用
        # ad采样频率 0x7d00=32000 007d 0000
        ad_frequency = [0x90, 0xd0, 0x03, 0x00]
        # 0x00013880 = 80000 , 0001应该对应0000的位置，且高低交换为0100 ，最终8038 0100
        # 0x0001d4c0 = 120000 , 最终c0d4 0100   ffffffff = 4294967295
        # 0x0003d090 = 250000 -> 90d0 0300
        ad_page = [0x00, 0x00, 0x00, 0x00]  # 程控放大倍数
        nopmiddle_1 = [0x00, 0x00, 0x00, 0x00]
        nopmiddle_2 = [0x00, 0x00, 0x00, 0x00]
        nopmiddle_3 = [0x00, 0x00, 0x00, 0x00]
        nopmiddle_4 = [0x00, 0x00, 0x00, 0x00]
        nopmiddle_5 = [0x00]
        ad_start_channel = [0x00]
        ad_end_channel = [0x08]
        mux_type = [0x00]  # 0-采用单端输入采集方式 ，1-双端输入采集方式
        nopbackward = [0x00, 0x00, 0x00, 0x00]
        setting = package_code + nopforward + ad_frequency + ad_page + nopmiddle_1 + nopmiddle_2 + \
            nopmiddle_3 + nopmiddle_4 + nopmiddle_5 + ad_start_channel + ad_end_channel + mux_type + nopbackward

        '''打包消息'''
        setting_message = struct.pack(
            "%dB" %
            (len(setting)),
            *
            setting)  # %d的位置对应后边列表的长度 ，结果是36B，表示可以存贮一个长度为36的无符号字符串
        # import math ; print("%.3fB" % math.pi) 表示一种格式化的写法，即以指定格式格式化，%
        # 用于连接被格式化的对象 , out = [3.142B]

        '''发送消息'''
        self.tcpClient.send(setting_message)

        '''接收返回结果'''
        setting_response = self.tcpClient.recv(64)

        '''解析返回结果'''
        setting_result = struct.unpack(
            "%dB" %
            (len(setting_response)),
            setting_response)  # 校验码 ，也就是发送时首个字段package_code
        # print([hex(setting_result[0]), hex(setting_result[1])]) #按字节返回十进制
        # ，hex转换成16进制
        if [setting_result[0], setting_result[1]] == [
                setting_message[0], setting_message[1]]:
            print('返回指令包与发送指令包相等，设置采集成功')
            return 1
        else:
            print("设置采集失败，请检查设置采集命令！")
            return 0

    def ADstop(self):
        '''停止采集指令结构表'''
        package_code = [0xa2, 0x5a]  # 指令符
        nop = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00]  # 占位符
        read = package_code + nop
        ##########################
        # 打包消息
        read_message = struct.pack("%dB" % (len(read)), *read)
        # 发送消息
        self.tcpClient.send(read_message)
        # 接收返回结果
        read_response = self.tcpClient.recv(1900)
        # print(read_response)  #b'\xa2Z\xce\xdc\x00\x00\x00\x00' 除了头2个字节不变其它都会随机变化
        # 解析返回结果
        read_result = struct.unpack(
            "%dB" %
            (len(read_response)),
            read_response)
        # (162, 90, 206, 220, 0, 0, 0, 0) = (a2, 5a, ce, dc, 00 , 00 ,00 ,00 ) 除了头2个字节不变其它都会随机变化
        return read_result

    def ADstart(self):
        '''启动采集指令结构表'''
        package_code = [0xa1, 0x5a]  # 启动采集指令码
        nop = [0x00, 0x00]  # 备用占位符
        ad_type = [0x00, 0x00, 0x00, 0x00]  # A/D 采集类型

        start = package_code + nop + ad_type
        ##########################
        # 打包消息
        start_message = struct.pack("%dB" % (len(start)), *start)
        # 发送消息
        self.tcpClient.send(start_message)
        # 接收返回结果
        start_response = self.tcpClient.recv(64)
        # 解析返回结果
        start_result = struct.unpack(
            "%dB" %
            (len(start_response)),
            start_response)
        # print([hex(start_result[0]), hex(start_result[1])])
        if [start_result[0], start_result[1]] == [
                start_message[0], start_message[1]]:
            print('返回指令包与发送指令包相等，启动采集成功')
            return 1
        else:
            print("启动采集失败，请检查启动采集命令！")
            return 0

    def ADDataRead(self):
        '''读采集数据指令结构表'''
        package_code = [0xa4, 0x5a]  # 停止采集指令码
        nop = [0x00, 0x00]  # 备用
        # 指定读取采集数据的个数  0—2^32 ,但是每次最多发送720个数据 02d0 = 720 ，这里指定接受 120000个数 c0d4
        # 0100 <-> 0001 d4c0
        ulLength = [0x98, 0x3a, 0x00, 0x00]

        read = package_code + nop + ulLength
        ##########################
        # 打包消息
        read_message = struct.pack("%dB" % (len(read)), *read)
        # 发送消息
        self.tcpClient.send(read_message)
        # 接收返回结果
        read_response = self.tcpClient.recv(1448 * 25)
        # 解析返回结果
        read_result = struct.unpack(
            "%dB" %
            (len(read_response)),
            read_response)
        # 删除头8个字节
        # read_result = read_result[8:-1] 这样会丢掉最后1个数
        # read_result = read_result[8:]
        k1 = np.arange(0, len(read_result), 1448)  # 0 ,1448,2896...28960
        k3 = []
        for i in read_result:
            k3.append(i)  # 先转换成列表就可以删除每1448个字节的头8个
        for i in k1:
            del k3[i:i + 8]  # 应当删除8*(len(read_result)/1448)  这么多个头8字节
        return k3, read_result, read_response

    def list_to_hex_to_str(self, List):
        '''将列表类型转化为连续的字符串'''
        Hex = [hex(i) for i in List]
        string = [str(j) for j in Hex]
        string = ''.join(string)
        string = string.replace('0x', '')
        return string

    def hex_to_signed_10(self, data):
        '''16进制 转换为有符号十进制'''
        width = 32  # 16进制数所占位数
        data = 'FFFF' + data
        dec_data = int(data, 16)
        if dec_data > 2 ** (width - 1) - 1:
            dec_data = 2 ** width - dec_data
            dec_data = 0 - dec_data
        dec_data = 5 * (-1 * dec_data / 3280.7) / 8  # 参考labview的缩放倍数
        return dec_data

    def hex_to_signed_10_much(self, string):
        '''string 应当是长度360的字符串'''
        String = []
        k = np.arange(0, len(string), 4)  # 0,4,8,...,352,356的等差数列
        k = k.astype(int)
        for i in k:
            temp = tcp.hex_to_signed_10(self, string[i:i + 4])
            String.append(temp)  # 返回的是列表类型
        return String

    def csv_tuple_str(self, temp_data):
        '''数据1440个字节 -> 每个通道180字节 -> 每个通道360个16进制数 -> 4个16进制也就是2个字节1个数,每个通道90个有符号十进制数'''
        '''所有通道 90 × 8 = 720 个数'''
        dict_tmp = {}
        Dict_tmp = {}
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
        Channel_0 = tcp.list_to_hex_to_str(self, channel_0)
        Channel_1 = tcp.list_to_hex_to_str(self, channel_1)
        Channel_2 = tcp.list_to_hex_to_str(self, channel_2)
        Channel_3 = tcp.list_to_hex_to_str(self, channel_3)
        Channel_4 = tcp.list_to_hex_to_str(self, channel_4)
        Channel_5 = tcp.list_to_hex_to_str(self, channel_5)
        Channel_6 = tcp.list_to_hex_to_str(self, channel_6)
        Channel_7 = tcp.list_to_hex_to_str(self, channel_7)
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
        Channel_0_10 = tcp.hex_to_signed_10_much(self, Channel_0)
        Channel_1_10 = tcp.hex_to_signed_10_much(self, Channel_1)
        Channel_2_10 = tcp.hex_to_signed_10_much(self, Channel_2)
        Channel_3_10 = tcp.hex_to_signed_10_much(self, Channel_3)
        Channel_4_10 = tcp.hex_to_signed_10_much(self, Channel_4)
        Channel_5_10 = tcp.hex_to_signed_10_much(self, Channel_5)
        Channel_6_10 = tcp.hex_to_signed_10_much(self, Channel_6)
        Channel_7_10 = tcp.hex_to_signed_10_much(self, Channel_7)
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
        # print(dec)
        return dt, dec, dict_tmp


class dataprocessing():
    def __init__(self, data=None):
        self.data = data
    # 函数1：判定当前时间序列是否是单调序列

    @staticmethod
    def mon(data):
        max_peaks = sig.argrelextrema(data, np.greater)[
            0]  # 极大值点横坐标，np.greater实现的是">"的功能
        min_peaks = sig.argrelextrema(data, np.less)[0]  # 极小值点横坐标
        num = len(max_peaks) + len(min_peaks)  # 极值点个数
        if num > 0:  # 不是单调序列
            return False
        else:
            return True

    # 函数2：寻找当前时间序列的极值点横坐标
    @staticmethod
    def findpeaks(data):
        return sig.argrelextrema(data, np.greater)[0]

    # 函数3：判断是否为IMF序列
    @staticmethod
    def imfyn(data):
        N = np.size(data)
        pzero = np.sum(data[0:N - 2] * data[1:N - 1] < 0)  # 过零点个数
        psum = np.size(dataprocessing.findpeaks(
            data)) + np.size(dataprocessing.findpeaks(-data))  # 极值点个数 这里引用时需要添加self否则报错
        if abs(pzero - psum) > 1:
            return False
        else:
            return True

    # 函数4：获取样条曲线
    @staticmethod
    def getspline(data):
        N = np.size(data)
        peaks = dataprocessing.findpeaks(data)
        peaks = np.concatenate(([0], peaks))
        peaks = np.concatenate((peaks, [N - 1]))
        if (len(peaks) <= 3):
            t = interpolate.splrep(
                peaks,
                y=data[peaks],
                w=None,
                xb=None,
                xe=None,
                k=len(peaks) - 1)
            return interpolate.splev(np.arange(N), t)
        t = interpolate.splrep(peaks, y=data[peaks])
        return interpolate.splev(np.arange(N), t)

    # 函数5：emd分解
    @staticmethod
    def emd(data):
        imf = []
        while not dataprocessing.mon(
                data):  # 只有emd需要传递_init_的参数，其它方法的输入参数并非原始数据，所以不能传递self.data
            x1 = data
            sd = np.inf
            while sd > 0.1 or (not dataprocessing.imfyn(x1)):
                s1 = dataprocessing.getspline(x1)
                s2 = -dataprocessing.getspline(-1 * x1)
                x2 = x1 - (s1 + s2) / 2
                sd = np.sum((x1 - x2) ** 2) / np.sum(x1 ** 2)
                x1 = x2
            imf.append(x1)
            data = data - x1
        imf.append(data)
        return imf

    # 函数6：模拟噪声函数wgn
    @staticmethod
    def wgn(data, snr):
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(data ** 2) / len(data)
        npower = xpower / snr
        return np.random.randn(len(data)) * np.sqrt(npower)

    # 函数7 ： 返回颜色16进制值
    @staticmethod
    def ReturnCmaps():
        cnames = {
            'aliceblue': '#F0F8FF',
            'antiquewhite': '#FAEBD7',
            'aqua': '#00FFFF',
            'aquamarine': '#7FFFD4',
            'azure': '#F0FFFF',
            'beige': '#F5F5DC',
            'bisque': '#FFE4C4',
            'black': '#000000',
            'blanchedalmond': '#FFEBCD',
            'blue': '#0000FF',
            'blueviolet': '#8A2BE2',
            'brown': '#A52A2A',
            'burlywood': '#DEB887',
            'cadetblue': '#5F9EA0',
            'chartreuse': '#7FFF00',
            'chocolate': '#D2691E',
            'coral': '#FF7F50',
            'cornflowerblue': '#6495ED',
            'cornsilk': '#FFF8DC',
            'crimson': '#DC143C',
            'cyan': '#00FFFF',
            'darkblue': '#00008B',
            'darkcyan': '#008B8B',
            'darkgoldenrod': '#B8860B',
            'darkgray': '#A9A9A9',
            'darkgreen': '#006400',
            'darkkhaki': '#BDB76B',
            'darkmagenta': '#8B008B',
            'darkolivegreen': '#556B2F',
            'darkorange': '#FF8C00',
            'darkorchid': '#9932CC',
            'darkred': '#8B0000',
            'darksalmon': '#E9967A',
            'darkseagreen': '#8FBC8F',
            'darkslateblue': '#483D8B',
            'darkslategray': '#2F4F4F',
            'darkturquoise': '#00CED1',
            'darkviolet': '#9400D3',
            'deeppink': '#FF1493',
            'deepskyblue': '#00BFFF',
            'dimgray': '#696969',
            'dodgerblue': '#1E90FF',
            'firebrick': '#B22222',
            'floralwhite': '#FFFAF0',
            'forestgreen': '#228B22',
            'fuchsia': '#FF00FF',
            'gainsboro': '#DCDCDC',
            'ghostwhite': '#F8F8FF',
            'gold': '#FFD700',
            'goldenrod': '#DAA520',
            'gray': '#808080',
            'green': '#008000',
            'greenyellow': '#ADFF2F',
            'honeydew': '#F0FFF0',
            'hotpink': '#FF69B4',
            'indianred': '#CD5C5C',
            'indigo': '#4B0082',
            'ivory': '#FFFFF0',
            'khaki': '#F0E68C',
            'lavender': '#E6E6FA',
            'lavenderblush': '#FFF0F5',
            'lawngreen': '#7CFC00',
            'lemonchiffon': '#FFFACD',
            'lightblue': '#ADD8E6',
            'lightcoral': '#F08080',
            'lightcyan': '#E0FFFF',
            'lightgoldenrodyellow': '#FAFAD2',
            'lightgreen': '#90EE90',
            'lightgray': '#D3D3D3',
            'lightpink': '#FFB6C1',
            'lightsalmon': '#FFA07A',
            'lightseagreen': '#20B2AA',
            'lightskyblue': '#87CEFA',
            'lightslategray': '#778899',
            'lightsteelblue': '#B0C4DE',
            'lightyellow': '#FFFFE0',
            'lime': '#00FF00',
            'limegreen': '#32CD32',
            'linen': '#FAF0E6',
            'magenta': '#FF00FF',
            'maroon': '#800000',
            'mediumaquamarine': '#66CDAA',
            'mediumblue': '#0000CD',
            'mediumorchid': '#BA55D3',
            'mediumpurple': '#9370DB',
            'mediumseagreen': '#3CB371',
            'mediumslateblue': '#7B68EE',
            'mediumspringgreen': '#00FA9A',
            'mediumturquoise': '#48D1CC',
            'mediumvioletred': '#C71585',
            'midnightblue': '#191970',
            'mintcream': '#F5FFFA',
            'mistyrose': '#FFE4E1',
            'moccasin': '#FFE4B5',
            'navajowhite': '#FFDEAD',
            'navy': '#000080',
            'oldlace': '#FDF5E6',
            'olive': '#808000',
            'olivedrab': '#6B8E23',
            'orange': '#FFA500',
            'orangered': '#FF4500',
            'orchid': '#DA70D6',
            'palegoldenrod': '#EEE8AA',
            'palegreen': '#98FB98',
            'paleturquoise': '#AFEEEE',
            'palevioletred': '#DB7093',
            'papayawhip': '#FFEFD5',
            'peachpuff': '#FFDAB9',
            'peru': '#CD853F',
            'pink': '#FFC0CB',
            'plum': '#DDA0DD',
            'powderblue': '#B0E0E6',
            'purple': '#800080',
            'red': '#FF0000',
            'rosybrown': '#BC8F8F',
            'royalblue': '#4169E1',
            'saddlebrown': '#8B4513',
            'salmon': '#FA8072',
            'sandybrown': '#FAA460',
            'seagreen': '#2E8B57',
            'seashell': '#FFF5EE',
            'sienna': '#A0522D',
            'silver': '#C0C0C0',
            'skyblue': '#87CEEB',
            'slateblue': '#6A5ACD',
            'slategray': '#708090',
            'snow': '#FFFAFA',
            'springgreen': '#00FF7F',
            'steelblue': '#4682B4',
            'tan': '#D2B48C',
            'teal': '#008080',
            'thistle': '#D8BFD8',
            'tomato': '#FF6347',
            'turquoise': '#40E0D0',
            'violet': '#EE82EE',
            'wheat': '#F5DEB3',
            'white': '#FFFFFF',
            'whitesmoke': '#F5F5F5',
            'yellow': '#FFFF00',
            'yellowgreen': '#9ACD32'}
        return cnames

    # 函数8：绘图函数plot
    @staticmethod
    def plot(
            x,
            y,
            legendname=None,
            titlename=None,
            xlabelname=None,
            ylabelname=None,
            Color='k'):
        '''
        :param x:  自变量
        :param y: 因变量
        :param legendname: 图例
        :param titlename: 标题
        :param xlabelname: 轴标签
        :param ylabelname: 轴标签
        :param Color: 颜色，如果不指定颜色 则随机选择颜色绘图
        :return: 图像
        '''
        '''设置图片大小，分辨率，[图像编号]，背景色，是否显示边框 ， 边框颜色'''
        '''线条类型
        '-'  |实线       '--'|虚线
        '-.  |虚点线     ':'|点线
        '.'  |点        ','|像素点
        'o'  |圆点      'v'|下三角点
        '^'  |上三角点   '<'|左三角点
        '>'  |右三角点   '1'|下三叉点
        '2'  |上三叉点   '3'|左三叉点
        '4'  |右三叉点   's'|正方点
        'p'  |五角点     '*'|星形点
        'h'  |六边形点1  'H'|六边形点2
        '+'  |加号点     'x'|乘号点
        'D'  |实心菱形点  'd'|瘦菱形点
        '_'  |横线点
        '''
        cnames = dataprocessing.ReturnCmaps()  # 得到color得16进制值
        if Color:
            color = Color
        else:
            color = random.sample(list(cnames.values()), 2)
        '''颜色类型 ：bgrcmykw ：蓝绿红青、品红、黄黑白'''
        plt.figure(
            figsize=(
                8,
                4),
            dpi=144,
            facecolor='w',
            frameon=True,
            edgecolor='b')

        '''设置图例文字，线颜色，线型，线宽，线风格，标记类型，标记尺寸，标记颜色，透明度'''
        plt.plot(
            x,
            y,
            label=legendname,
            color=color[0],
            linewidth=0.6,
            linestyle='--',
            marker='o',
            markersize='2',
            markerfacecolor=color[0],
            alpha=0.8)

        '''设置图例的文字，位置 ， 文字大小， 多图例按列数显示(默认按行)， 图例标记为原尺寸的多少倍大小 ， 是否添加阴影 ， 图里圆角方角，是否带边框， 边框透明度'''
        plt.legend(
            (legendname,
             ),
            loc='upper right',
            fontsize=18,
            ncol=1,
            markerscale=1,
            shadow=False,
            fancybox=True,
            frameon=True,
            framealpha=0.5)

        '''设置网格线是否出现、显示哪个轴的网格线 ，网格线颜色， 风格、 宽度、 设置次刻度线（default = major）'''
        plt.grid(
            b=None,
            axis='x',
            color='c',
            linestyle='-',
            linewidth=8,
            which='major')

        '''设置坐标轴范围'''
        ymax = max(y)
        ymin = min(y)
        xmax = max(x)
        xmin = min(x)
        plt.ylim(ymin, ymax)
        plt.xlim(xmin, xmax)

        '''设置轴标签是否旋转，文字垂直和水平位置'''
        plt.xlabel(xlabelname, fontsize=18, rotation=None)
        plt.ylabel(ylabelname, fontsize=18, rotation=90)

        '''设置轴刻度'''
        plt.yticks(())
        plt.xticks(())

        '''设置哪个轴的属性、次刻度线设置， 刻度线显示方式(绘图区内测、外侧、同时显示)，刻度线宽度，刻度线颜色、刻度线标签颜色(任命/日期等)
        刻度线与标签距离、刻度线标签大小、刻度线是否显示(default=bottom ,left)、刻度线标签是否显示(default = bottom/left)'''
        plt.tick_params(
            axis='both',
            which='major',
            direction='in',
            width=1,
            length=3,
            color='k',
            labelcolor='k',
            pad=1,
            labelsize=15,
            bottom=True,
            right=True,
            top=False,
            labeltop=False,
            labelbottom=True,
            labelleft=True,
            labelright=False)

        '''方框外形、背景色、 透明度、方框粗细、方框到文字的距离'''
        bb = dict(boxstyle='round', fc='w', ec='m', alpha=0.8, lw=10, pad=1.2)

        '''设置标题文字大小、标题大小、标题正常/斜体/倾斜、垂直位置、水平位置、透明度、标题背景色、是否旋转、标题边框有关设置（字典格式）'''
        plt.title(
            label=titlename,
            fontsize=20,
            fontweight='normal',
            fontstyle='italic',
            alpha=0.8,
            backgroundcolor='w',
            rotation=None)  # bbox = bb

        plt.show()

    # 函数9 数据归一化及反归一化函数
    @staticmethod
    def MaxMinNormalized(
            array,
            mode=0,
            reverse=False,
            col_max=None,
            col_min=None,
            row_max=None,
            row_min=None):
        '''
        :param array: 输入为数组格式
        :param mode: mode=0表示按行归一化,mode=1表示按列归一化
        按行归一化,需要得到每列的最值,然后某一列的每行数据都用这个最大最小值归一化
        :param reverse: bool 反归一化
        :param col_max: 按行反归一化时需要原始数据每行的最大值
        :param col_min: 按行反归一化时需要原始数据每行的最小值
        :param row_max: 按列反归一化时需要原始数据每行的最大值
        :param row_min: 按列反归一化时需要原始数据每行的最小值
        :return: 归一化或反归一化后的数组
        '''
        data_shape = array.shape  # 返回数组的行列数
        data_rows = data_shape[0]  # 行数
        data_cols = data_shape[1]  # 列数
        norm = np.empty((data_rows, data_cols))  # 用于存放归一化后的数据
        if not reverse:
            if mode == 0:  # 选择按行归一化模式
                maxcols = array.max(axis=0)  # 返回第0列的最大值，即每列的最大值，返回行向量
                mincols = array.min(axis=0)
                for i in range(data_cols):  # 循环对每行依次处理 ,单列依次对行处理
                    # 每行应用公式 (x - xmin) / (xmax - xmin),[-1,1]之间则是2*(x - xmin)
                    # / (xmax - xmin)-1
                    norm[:, i] = (2 * (array[:, i] - mincols[i]) /
                                  (maxcols[i] - mincols[i])) - 1
                return norm
            if mode == 1:  # 选择按列归一化模式
                maxrows = array.max(axis=1)  # 返回每行的最大值
                minrows = array.min(axis=1)
                for i in range(data_rows):  # 循环对每列依次处理 ，单行对每列依次处理
                    norm[i, :] = (2 * (array[i, :] - minrows[i]) /
                                  (maxrows[i] - minrows[i])) - 1
                return norm
        elif reverse:  # reverse 为True执行下述操作
            if mode == 0:
                maxcols = array.max(axis=0)
                mincols = array.min(axis=0)
                for i in range(data_cols):
                    norm[:, i] = ((array[:, i] + 1) / 2) * (col_max[i] - col_min[i]) + col_min[
                        i]  # 原数组每列的最大值和最小值 x_norm *(xmax-xmin)+xmin
                # 或 (x_norm+1)/2 *(xmax-xmin) + xmin
                return norm
            if mode == 1:
                maxrows = array.max(axis=1)
                minrows = array.min(axis=1)
                for i in range(data_rows):
                    norm[i, :] = ((array[:, i] + 1) / 2) * \
                        (row_max[i] - row_min[i]) + row_min[i]
                return norm

    # 函数10 求阈值函数
    @staticmethod
    def Thr(x):
        len1 = len(x)
        w = sorted(x)  # 从小到大排序
        if len1 % 2 == 1:
            v = w[int((len1 + 1) / 2) - 1]  # 如len1=5，计算结果为3 ，但是3不是中间位置还需要减1
        else:
            v = (w[int(len1 / 2) - 1] + w[int(len1 / 2)]) / \
                2  # 是 偶数就 取中间两个值的平均值
        sigmal = abs(v) / 0.6745
        valve = sigmal * ((2 * (math.log(len1, math.e))) ** (0.5))
        return valve

    # 函数11 matlab去噪函数
    @staticmethod
    def Matlab_Quzao(data):
        '''输入的数据是行向量，然后首先得到最大分解水平，然后wavedec函数得到分解后的近似系数和细节系数'''
        '''然后外循环是每层细节系数，内循环是对每层系数进行内层处理，求取每层细节系数的阈值'''
        mode = 'db8'
        w = pywt.Wavelet(mode)
        maxlev = pywt.dwt_max_level(len(data), w.dec_len)  # 找到数据可以分解的最大水平
        # maxlev = 3 #一般可以取3
        # 这是可以分解的最大水平，但是不推荐，本信号最大可分解6层，但是3 最好
        print(f"matlab去噪函数 - 原始数据在{mode}小波基下的最大分解水平为：" + str(maxlev))
        # 返回高频近似系数和细节系数，#返回CAn,CDn,CDn-1,CDn-2,....CD1
        coeffs = pywt.wavedec(data, w, level=int(maxlev / 2))
        for j in range(1, len(coeffs)):  # 外循环， 不同的细节系数轮流去噪 ,从1开始，因为第1个是近似系数
            # print(coeffs[j])
            valve = 5
            # valve = dataprocessing.Thr(coeffs[j])  # 当前的细节系数 计算阈值
            # print(valve)
            temp = coeffs[j]  # 把当前细节系数给副本temp ，整行赋值
            for i in range(
                    int(len(temp))):  # 内循环 ，对当前细节系数每一个元素进行处理 ， 循环次数取决于 当前细节系数的长度
                if (abs(temp[int(i)]) <= valve):  # 绝对值小于等于阈值的都取0
                    temp[int(i)] = 0
                else:
                    if (temp[int(i)] > valve):
                        temp[int(i)] = temp[int(i) - 1] - \
                            valve  # 绝对值大于阈值的都减去阈值
                    else:
                        temp[int(i)] = temp[int(i) - 1] + \
                            valve  # 其他情况 ，比如很大的负数，加上阈值
            coeffs[j] = temp  # 可以这样赋值，把当前去噪后的细节系数替换掉 存放细节系数的coeffs
        datarec = pywt.waverec(coeffs, w)  # 外循环结束 ，  所有替换过的 coeffs 再做逆重构
        return datarec

    # 函数12 python 的pywavlet去噪函数
    @staticmethod
    def Python_Quzao(x):
        '''输入数据是行向量，返回也是行向量'''
        mode = 'db4'  # 选择何种小波基
        w = pywt.Wavelet(mode)
        maxlev = pywt.dwt_max_level(len(x), w.dec_len)
        print(f"python去噪函数 - 原始数据在{mode}小波基的最大分解水平为： " + str(maxlev))
        threshold = 0.02
        coeffs = pywt.wavedec(x, mode, level=maxlev)
        for i in range(1, len(coeffs)):  # 只对细节系数处理 ，也就是高频分量去噪
            # print(coeffs[i])
            # print(coeffs[i].max())
            # 将max(coeffs[i]) 改为coeffs[i].max()
            coeffs[i] = pywt.threshold(coeffs[i], threshold * coeffs[i].max())
        datarec = pywt.waverec(coeffs, mode)  # 将信号小波重构
        return datarec

    # 函数13 获取样本的缺失值
    @staticmethod
    def draw_missing_data_table(df):
        '''
        1、isnull 用于依次判定每个元素是否为空值，如果是返回True ，属于bool变量
        2、sum 计算 每列缺失值的个数
        3、sort_values 按每列缺失值的个数进行排序，ascending表示升序，即从小到大
        4、count 返回序列的长度，或者说每列单元格个数
        '''
        '''
        测试程序:
        D = np.random.randn(1300,3)
        D.iloc[2:8,0] = np.nan
        D.iloc[2:14,1] = np.nan
        D.iloc[2:6,2] = np.nan
        print(D.isnull().sum().sort_values(ascending=False))
        print((D.isnull().sum() / D.isnull().count()).sort_values(ascending=True))
        print(draw_missing_data_table(D)) # 第1列有12个缺失值，第0列6个，第2列5个
        '''
        total = df.isnull().sum().sort_values(
            ascending=False)  # 返回每列包含的缺失值的个数，并按照值的大小进行降序
        percent = (
            df.isnull().sum() /
            df.isnull().count()).sort_values(
            ascending=False)
        missing_data = pd.concat(
            [total, percent], axis=1, keys=['Total', 'Percent'])
        return missing_data

    # 函数14 Z-Score标准化归函数
    @staticmethod
    def NormDetection(dataframe):
        '''
        输入数据必须是dataframe格式，如果是数组格式会自动转换为dataframe格式
        零-均值规范化也称标准差标准化，经过处理的数据的均值为0，标准差为1。转化公式为：
        (x-xmean)/xstd
        '''
        norm = pd.DataFrame()
        if type(dataframe) != type(norm):
            dataframe = pd.DataFrame(dataframe)
        mean_row = dataframe.mean(axis=0)  # 返回每列的均值
        std_row = dataframe.std(axis=0)  # 返回每列的均值
        for row, col in dataframe.iterrows():
            temp = (col - mean_row) / std_row  # 用每列的值减去对应列的均值再除以该列的标准差
            # 将每列标准化的值返回 (x-xmean)/xstd
            norm = pd.concat([norm, temp], axis=1, ignore_index=True)
        norm = norm.T  # 到此为止相当于得到了dataframe每个值的标准化
        # 后续人为观察 数据是否存在异常
        return norm

    # 函数15 计算程序用时
    @staticmethod
    def pass_time(start_time, end_time):
        sum_time = math.floor(end_time - start_time)
        h = math.floor(sum_time / 3600)
        m = math.floor((sum_time - h * 60 * 60) / 60)
        s = (sum_time - h * 60 * 60 - m * 60)
        print('\n用时时间:')
        print('hour:{0}  minute:{1}  second:{2}'.format(h, m, s))

    # 函数16 进度条函数
    @staticmethod
    def ProgressBar(start=None, mode=False):
        '''mode默认为假，执行下属程序；如果mode=True只输出执行开始'''
        try:
            scale = 50
            if not mode:  # mode为假或者默认会执行下属程序，如果mode指定True则只执行第一句
                # print("执行开始，祈祷不报错".center(scale // 2, "-"))
                # start = time.perf_counter()
                for i in range(scale + 1):
                    a = "*" * i
                    b = "." * (scale - i)
                    c = (i / scale) * 100
                    dur = time.perf_counter() - start
                    print(
                        "\r{:^3.0f}%[{}->{}]{:.2f}s".format(c, a, b, dur), end="")
                    time.sleep(0.1)
                print("\n" + "执行结束，万幸".center(scale // 2, "-"))
            else:
                print("执行开始，祈祷不报错".center(scale // 2, "-"))
        except BaseException:
            print('不好意思，程序有BUG，您需要重新调试！')

    # 函数17 RGB值梯度生成
    @staticmethod
    def plot_color_gradients(cmap_category, cmap_list, nrows, gradient):
        fig, axes = plt.subplots(nrows=nrows)
        fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99)
        axes[0].set_title(cmap_category + ' colormaps', fontsize=14)

        for ax, name in zip(axes, cmap_list):
            ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(name))
            pos = list(ax.get_position().bounds)
            x_text = pos[0] - 0.01
            y_text = pos[1] + pos[3] / 2.
            fig.text(
                x_text,
                y_text,
                name,
                va='center',
                ha='right',
                fontsize=10)

        # Turn off *all* ticks & spines, not just the ones with colormaps.
        for ax in axes:
            ax.set_axis_off()

    # 函数18 显示可用颜色函数
    @staticmethod
    def showcolor():
        # Have colormaps separated into categories:
        # http://matplotlib.org/examples/color/colormaps_reference.html
        cmaps = [('Perceptually Uniform Sequential', [
            'viridis', 'plasma', 'inferno', 'magma']),
            ('Sequential', [
                'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']),
            ('Sequential (2)', [
                'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
                'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
                'hot', 'afmhot', 'gist_heat', 'copper']),
            ('Diverging', [
                'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
                'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']),
            ('Qualitative', [
                'Pastel1', 'Pastel2', 'Paired', 'Accent',
                'Dark2', 'Set1', 'Set2', 'Set3',
                'tab10', 'tab20', 'tab20b', 'tab20c']),
            ('Miscellaneous', [
                'flag', 'prism', 'ocean', 'gist_earth', 'terrain', 'gist_stern',
                'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg', 'hsv',
                'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar'])]

        nrows = max(len(cmap_list) for cmap_category, cmap_list in cmaps)
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient))
        for cmap_category, cmap_list in cmaps:
            dataprocessing.plot_color_gradients(
                cmap_category, cmap_list, nrows, gradient)
        plt.show()

    # 函数19 分类绘制散点图函数
    @staticmethod
    def ClassificationDrawing(
            DataFrame,
            DataFrame_x_name,
            DataFrame_y_name,
            DataFrame_class_name,
            mode='scatter'):
        '''
        :param DataFrame: 输入数据格式为DataFrame
        :param DataFrame_x_name: DataFrame_x_name,DataFrame_y_name 选择哪两列数据进行绘图，字符串格式
        :param DataFrame_y_name:
        :param DataFrame_class_name: DataFrame_class_name 为DataFrame 类的划分，字符串格式
        :param mode: mode 为采取哪种绘图方式，默认scatter ,可选plot ，cmaps
                     mode 可选 scatter 和 plot 2种方式 ， 字符串
        :return: 都定义为None 值是为了方便传递出cmaps值 ，用于plot函数的使用
        '''
        data = DataFrame  # 简记
        DataFrame_class = DataFrame[DataFrame_class_name]  # 找到该列
        Class = DataFrame_class.unique()  # 找到class所有的类别
        modules = [[] for _ in range(len(Class))]  # 用于存放各类，二维列表
        for i in DataFrame_class:  # i是字符串
            for j in range(len(Class)):
                if i == Class[j]:  # 检测 i 属于 哪一类 ，如果相等则赋予modules对应的单个空列表
                    modules[j] = data.loc[(data[DataFrame_class_name] == i), [
                        DataFrame_x_name, DataFrame_y_name]]  # 找到符合条件的两列
        # colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']  # 常见的七种颜色
        markers = [
            'x',
            '*',
            'h',
            'o',
            '<',
            '>',
            '^',
            'v',
            'd',
            'D',
            'H',
            '+',
            's',
            'p',
            '|',
            '.',
            '_',
            '1',
            '2',
            '3',
            '4',
            '8']  # 常见的标记
        cmaps = [
            'coolwarm',
            'plasma',
            'Oranges',
            'summer',
            'Spectral',
            'Set1',
            'rainbow',
            'magma',
            'inferno',
            'viridis',
            'PRGn',
            'PiYG',
            'coolwarm_r',
            'bwr_r',
            'bone',
            'gray',
            'pink',
            'hot',
            'summer_r',
            'autumn_r',
            'spring_r',
            'winter_r',
            'cool',
            'Greens',
            'copper',
            'Reds',
            'Oranges',
            'OrRd',
            'PuBu',
            'tab10',
            'tab20',
            'tab20b',
            'tab20c',
            'Pastel1',
            'Pastel2',
            'Set2',
            'Set3',
            'Accent',
            'Dark2']
        cnames = dataprocessing.ReturnCmaps()
        plt.figure(
            figsize=(
                8,
                6.18),
            dpi=144,
            facecolor='w',
            frameon=True,
            edgecolor='b')
        if mode == 'scatter':
            for k in range(len(modules)):
                temp = modules[k]
                color = random.sample(
                    list(
                        cnames.values()),
                    len(modules))  # 字典类型取值不是字符串，需要先转换为字符串
                # color = random.sample(colors,len(modules)) #随机选颜色和标记
                marker = random.sample(markers, len(modules))
                cmap = random.sample(cmaps, len(modules))
                plt.scatter(temp.iloc[:,
                                      0],
                            temp.iloc[:,
                                      1],
                            color=color[k],
                            s=80,
                            cmap=cmap,
                            label=Class[k],
                            marker=marker[k],
                            linewidths=0.6,
                            alpha=0.8)
            plt.legend()
            plt.yticks(())
            plt.xticks(())
            plt.show()
        if mode == 'plot':
            dataprocessing.plot(
                data[DataFrame_x_name],
                data[DataFrame_y_name])  # 此时颜色列表不起作用
        # return cnames

    # 函数20 sns.pairplot(多变量关系图)
    @staticmethod
    def RelationChart(
            dataframe,
            hue=None,
            palette='husl',
            markers=None,
            diag_kind='auto',
            kind='scatter',
            vars=None,
            x_vars=None,
            y_vars=None,
            corner=False,
            height=2.5,
            aspect=1):
        '''
        :param dataframe: pd格式的输入
        :param hue: 按类绘图
        :param palette: 使用调色板
        :param markers: 使用不同标记
        :param diag_kind: 可选'auto','hist','kde' 对角线图样式
        :param kind:可选'scatter','kde' 非对角线图样式
        :param vars: 如果不指定，默认使用所有列进行分析，或者指定两列，如vars=["Number of times pregnant","Age (years)"]
        :param x_vars:用x_vars和 y_vars参数指定,x_vars和y_vars要同时指定
        :param y_vars:指定的x_vars中的每一个会与y_vars中的每一个进行配对
        :param corner:默认False，Ture时只显示下三角形
        :param height: 图高度
        :param aspect: 图宽度
        :return:plotgrid
        '''
        sns.pairplot(
            dataframe,
            hue=hue,
            palette=palette,
            markers=markers,
            diag_kind=diag_kind,
            kind=kind,
            vars=vars,
            corner=corner,
            x_vars=x_vars,
            y_vars=y_vars,
            height=height,
            aspect=aspect)
        plt.show()

    # 函数21 类别重编码函数

    @staticmethod
    def LabelCode(dataframe, label):
        '''
        :param dataframe: 数据格式为dataframe
        :param label: 需要被重新编码的类别的列名字，格式为字符串
        :return: 返回编码好的dataframe 以及数组格式的标签
        '''
        '''
        测试程序：
        from sklearn.preprocessing import LabelEncoder
        from pandas import DataFrame
        data = {'color':['green','red','blue'],
                'size':['M','L','XL'],
                'price':['10.1','13.5','15.3'],
               'classlabel':['class1','class2','class1']}
        df = DataFrame(data)
        from Car import Troubleshooting
        tuple = Troubleshooting.dataprocessing.LabelCode(df,'classlabel')
        '''
        class_le = LabelEncoder()  # 初始化一个类
        y = class_le.fit_transform(dataframe[label].values)  # 把类别转化为相应整型
        class_le.inverse_transform(y)  # 逆转化
        del dataframe[label]
        dataframe[label] = y
        return dataframe, y

    @staticmethod
    # 函数22 占位符函数
    def Pass():
        pass

    @staticmethod
    # 函数23 五点m次平滑法去噪
    # @jit
    def Mean_3_5(Series, m, Color=None):
        '''
        :param Series: 时间序列，列向量 ，array格式
        :param m: 迭代次数，一般取3
        :param Color: 颜色可指定，字符 ； 不指定时随机颜色
        :return: 去噪前后的比对，以及去噪后的序列
        '''
        n = len(Series)
        a = Series
        b = Series.copy()
        for i in range(m):
            b[0] = (69 * a[0] + 4 * a[1] - 6 * a[2] + 4 * a[3] - a[4]) / 70
            b[1] = (2 * a[0] + 27 * a[1] + 12 * a[2] -
                    8 * a[3] + 2 * a[4]) / 30  # 35
            for j in range(2, n - 2):
                b[j] = (-3 * a[j - 2] + 12 * a[j - 1] + 17 *
                        a[j] + 12 * a[j + 1] - 3 * a[j + 2]) / 35
            b[n - 2] = (2 * a[n - 5] - 8 * a[n - 4] + 12 *
                        a[n - 3] + 27 * a[n - 2] + 2 * a[n - 1]) / 35
            b[n - 1] = (-a[n - 5] + 4 * a[n - 4] - 6 * a[n - 3] +
                        4 * a[n - 2] + 69 * a[n - 1]) / 70
            a = b.copy()
        #dataprocessing.plot(np.arange(0,n,1),Series,legendname='Before Denoising',Color=Color)
        #dataprocessing.plot(np.arange(0,n,1),a, legendname='After Denoising',Color=Color)
        return a

    @staticmethod
    # 函数24 寻找极值及其横坐标函数
    def FindExtremum(Series):
        '''
        :param Series: 时间序列
        :return: index 为极大值、极小值的横坐标 x_extremum 对应的极大值和极小值
        相当于可以返回 t1、t3、t2 和 i1、i3、i2 (i4是合闸电流的结束时间，单独进行处理)
        '''
        x = Series
        index1 = signal.argrelextrema(x, np.greater_equal)  # 返回极大值点的横坐标 元组形式
        index2 = signal.argrelextrema(-x, np.greater_equal)  # 极小值的横坐标
        x_extremum_max = x[index1]  # 对应的最大值
        x_extremum_min = x[index2]  # 对应的最小值
        index = np.concatenate((index1, index2), axis=1)  # axis=1表示横向拼接
        x_extremum = np.concatenate((x_extremum_max, x_extremum_min), axis=0)
        index = np.array(index)
        x_extremum = np.array(x_extremum)
        plt.figure(figsize=(10, 6.18))
        plt.plot(np.arange(len(x)), x)
        plt.plot(signal.argrelextrema(x, np.greater_equal)[
                 0], x[signal.argrelextrema(x, np.greater_equal)], 'o', markersize=20)
        plt.plot(signal.argrelextrema(-x, np.greater_equal)
                 [0], x[signal.argrelextrema(-x, np.greater_equal)], '*', markersize=20)
        plt.show()
        return index, x_extremum

    @staticmethod
    # 函数25 计算时间序列的相关统计量
    def Statistics(Series):
        x = Series
        x_peak = np.max(abs(x))  # 1、峰值
        x_mean = np.mean(x)  # 2、均值
        x_mean_f = sum(abs(x)) / len(x)  # 3、平均幅值
        x_std = np.std(x)  # 4、标准差
        # 5、方差 #sum(pow((x-np.mean(x)),2))/(len(x)-1)
        x_var = np.var(x, ddof=1)
        # 自由度 ddof ： 默认是0，也就是被除数为(N-ddof) ，取1时(N-1)
        x_rms = math.sqrt(pow(x_mean, 2) + pow(x_std, 2))  # 6、均方根
        x_skew = np.mean((x - x_mean) ** 3) / pow(x_std, 3)  # 7、偏度
        x_kurt = np.mean((x - x_mean) ** 4) / pow(np.var(x), 2)  # 8、峭度
        x_max = np.max(x)  # 9、最大值
        x_min = np.min(x)  # 10、最小值

        def rmsm(x):
            x_rmsm = 0
            for i in range(len(x)):
                temp = math.sqrt(abs(x[i]))
                x_rmsm = x_rmsm + temp
            x_rmsm = pow(x_rmsm / len(x), 2)
            return x_rmsm
        x_rmsm = rmsm(x)  # 11、均方根幅值
        x_ydz = x_peak / x_rmsm  # 12、裕度指标
        x_bxz = x_rms / x_mean_f  # 13、波形指标
        x_mcz = x_peak / x_mean  # 14、脉冲指标
        x_fzz = x_peak / x_rms  # 15、峰值指标
        x_qdz = x_kurt / x_rms  # 16、峭度指标
        Features = np.array(
            (x_peak,
             x_mean,
             x_mean_f,
             x_std,
             x_var,
             x_rms,
             x_skew,
             x_kurt,
             x_max,
             x_min,
             x_rmsm,
             x_ydz,
             x_bxz,
             x_mcz,
             x_fzz,
             x_qdz))
        return Features

    @staticmethod
    # 函数26 饼图绘制
    def PythonPie():
        '''
        x       :(每一块)的比例，如果sum(x) > 1会使用sum(x)归一化；
        labels  :(每一块)饼图外侧显示的说明文字；
        explode :(每一块)离开中心距离；
        startangle :起始绘制角度,默认图是从x轴正方向逆时针画起,如设定=90则从y轴正方向画起；
        shadow  :在饼图下面画一个阴影。默认值：False，即不画阴影；
        labeldistance :label标记的绘制位置,相对于半径的比例，默认值为1.1, 如<1则绘制在饼图内侧；
        autopct :控制饼图内百分比设置,可以使用format字符串或者format function
                '%1.1f'指小数点前后位数(没有用空格补齐)；
        pctdistance :类似于labeldistance,指定autopct的位置刻度,默认值为0.6；
        radius  :控制饼图半径，默认值为1；
        counterclock ：指定指针方向；布尔值，可选参数，默认为：True，即逆时针。将值改为False即可改为顺时针。
        wedgeprops ：字典类型，可选参数，默认值：None。参数字典传递给wedge对象用来画一个饼图。例如：wedgeprops={'linewidth':3}设置wedge线宽为3。
        textprops ：设置标签（labels）和比例文字的格式；字典类型，可选参数，默认值为：None。传递给text对象的字典参数。
        center ：浮点类型的列表，可选参数，默认值：(0,0)。图标中心位置。
        frame ：布尔类型，可选参数，默认值：False。如果是true，绘制带有表的轴框架。
        rotatelabels ：布尔类型，可选参数，默认为：False。如果为True，旋转每个label到指定的角度。
        '''
        labels = '面向对象', '丰富的库', '解释型语言', '简单易学', '可拓展性'  # 自定义标签
        sizes = [20, 20, 20, 20, 20]  # 每个标签占多大
        explode = (0, 0, 0, 0, 0)  # 将某部分爆炸出来
        plt.pie(
            sizes,
            explode=explode,
            labels=labels,
            autopct=None,
            shadow=False,
            startangle=90)
        # autopct，圆里面的文本格式，%1.1f%%表示小数有1位，整数有一位的浮点数
        # shadow，饼是否有阴影
        # startangle，起始角度，0，表示从0开始逆时针转，为第一块。一般选择从90度开始比较好看
        plt.axis('equal')
        plt.title('Python的特点')
        plt.show()

    @staticmethod
    # 函数27 字符串提取数字函数
    def StrextractDigit(str):
        num = []
        for i in range(len(str)):
            # print(mystr[i].isdigit())
            checked = str[i].isdigit()
            if checked:
                num.append(str[i])
        num = ''.join(num)  # 在列表的每一个元素加入指定对象拼接
        return num

    @staticmethod
    # 函数28 DataFrame行列命名函数
    def ColumName(name, length):
        # columnnames = ColumName("energy",32)
        namelist = [[] for _ in range(length)]
        temp = ([str(x) for x in range(length)])
        for i in range(length):
            namelist[i] = name + '_' + temp[i]
        return namelist

    @staticmethod
    # 函数29 DataFrame去除NAN值
    def nan_to_zero(DataFrame):
        for i in range(DataFrame.shape[1]):
            for j in range(DataFrame.shape[0]):
                if pd.isnull(DataFrame.iloc[j, i]):
                    DataFrame.iloc[j, i] = 0
        return DataFrame

    @staticmethod
    # 函数30 循环处理合并大量csv文件的函数
    def contact_csv(filename):
        # filename = "csv_线圈不正常"
        address = "C:/Users/chenbei/Desktop/钢/数据/"
        address = address + filename
        temp = ([str(x) for x in range(101)])
        Address = [[] for _ in range(101)]
        # Data = pd.DataFrame(columns=[str(x) for x in range(101)])
        Data = pd.DataFrame()
        for i in range(101):
            Address[i] = address + "/" + temp[i] + ".csv"
            data = pd.read_csv(Address[i])
            data.columns = ['0', 'data']
            del data['0']
            data.columns = [str(i)]
            Data = pd.concat([Data, data], axis=1, ignore_index=True)
        return Data


class feature_extract():
    '''特征提取类'''

    def __init__(self, data):
        self.data = data

    @staticmethod
    # 函数1 获取样本矩阵的小波特征向量
    def WaveletAlternation(SingleSample_Data):
        '''输入数组格式 ： 例如2043 * 100 表示同一种故障的信号采集了100组，每组长度2043'''
        '''
        测试程序：
        D = pd.read_csv(r"C:/Users\chenbei\Desktop\钢\D.csv")
        k = WaveletAlternation(D.values)
        '''
        Featureweidu, SingleDir_Samples = SingleSample_Data.shape  # 获取矩阵的行数和列数，即样本维数 2043 * 100
        # 定义样本特征向量 #Array 形式 100 * 8 ，每组行向量是原始信号每列的特征
        SingleDir_SamplesFeature = np.zeros((SingleDir_Samples, 8))
        #      SingleDir_SamplesFeature = [] # list形式
        for i in range(SingleDir_Samples):
            SingleSampleDataWavelet = SingleSample_Data[:, i]  # 对第i列做小波包分解
            # 进行小波变换，提取样本特征
            wp = pywt.WaveletPacket(
                SingleSampleDataWavelet,
                wavelet='db3',
                mode='symmetric',
                maxlevel=3)  # 小波包三层分解
            #            print([node.path for node in wp.get_level(3, 'natural')])   #第3层有8个
            # 获取第level层的节点系数
            aaa = wp['aaa'].data  # 第1个节点
            aad = wp['aad'].data  # 第2个节点
            ada = wp['ada'].data  # 第3个节点
            add = wp['add'].data  # 第4个节点
            daa = wp['daa'].data  # 第5个节点
            dad = wp['dad'].data  # 第6个节点
            dda = wp['dda'].data  # 第7个节点
            ddd = wp['ddd'].data  # 第8个节点
            # 求取节点的范数
            ret1 = np.linalg.norm(aaa, ord=None)  # 第一个节点系数求得的范数/ 矩阵元素平方和开方
            ret2 = np.linalg.norm(aad, ord=None)
            ret3 = np.linalg.norm(ada, ord=None)
            ret4 = np.linalg.norm(add, ord=None)
            ret5 = np.linalg.norm(daa, ord=None)
            ret6 = np.linalg.norm(dad, ord=None)
            ret7 = np.linalg.norm(dda, ord=None)
            ret8 = np.linalg.norm(ddd, ord=None)
            # 8个节点组合成特征向量
            SingleSampleFeature = [
                ret1, ret2, ret3, ret4, ret5, ret6, ret7, ret8]
            SingleDir_SamplesFeature[i][:] = SingleSampleFeature  # Array 形式
        #            SingleDir_SamplesFeature.append(SingleSampleFeature)   #list 形式
        #            print('SingleDir_SamplesFeature:', SingleDir_SamplesFeature)
        return SingleDir_SamplesFeature  # 返回的是1×8的特征向量，如果是100组则 100×8

    # 函数2 函数1的增强版本，除了返回特征值以外 ，还可以返回多小波包能量矩的横向与纵向分布，默认横向

    @staticmethod
    def WaveletAlternationEnhance(
            Signals,
            name,
            maxlevel,
            tree=False,
            plot=False,
            mode=1,
            which=1):
        '''
        :param Signals: 输入格式dataframe,必须使用不为数字的列名,序列长度×样本数 , 如1300×100
        :param name: string , 选取的小波基名称
        :param maxlevel: 最大分解水平,一般取3,4,5 对应节点数 8 ,16 ,32
        :param tree: bool 是否绘制小波序列
        :param plot: bool 是否绘制能量矩
        :param mode: int  0 or 1 , 默认横向分布直方图
        :param which: int plot有效且mode=1时有效,选择堆stack还是kde模式
        :return: 小波包特征量和能量矩分布图
        '''
        '''
        测试程序
        D = pd.read_csv(r"C:/Users\chenbei\Desktop\钢\D.csv")
        Feature  = WaveletAlternationEnhance(D,'db5',5,plot= True ,which=0)
        '''
        # 防止后边列名和行名默认数字排列，这里先行将行列取好名字
        columnnames = dataprocessing.ColumName("sample", Signals.shape[1])
        indexnames = dataprocessing.ColumName("node", Signals.shape[0])
        Signals.columns = columnnames
        Signals.index = indexnames
        # pywt.families() #打印小波家族
        signals = Signals.values
        rows, cols = signals.shape
        modenum = pow(2, maxlevel)  # 节点个数
        Feature = np.zeros((cols, modenum))  # 存放特征量,level=3时 有8个特征量
        for i in range(cols):
            test_data = signals[:, i]  # 第i列信号用于小波分解
            wp = pywt.WaveletPacket(
                data=test_data,
                wavelet=name,
                maxlevel=maxlevel,
                mode='symmetric')  # 某列数据的小波包实例化对象
            leafmode = [node.path for node in wp.get_leaf_nodes(
                decompose=True)]  # 获取全部的叶子节点名称,数量为2^maxlevel
            # 存放节点得时间序列值 , 指定长度len(leafmode)
            nodelist = [[] for k in range(len(leafmode))]
            dict_temp = dict.fromkeys(leafmode)  # 存放所有节点系数的字典,keys值为节点名称
            for j in range(len(leafmode)):
                nodelist[j] = wp[leafmode[j]].data  # 把叶子节点时间序列值赋给存储列表
            for j in range(len(leafmode)):
                dict_temp[leafmode[j]] = np.linalg.norm(
                    nodelist[j], ord=None)  # 循环找到每个叶子节点的时间序列值并求范数,对应的值赋给字典相应节点键
            list_temp = list(dict_temp.values())  # 字典值只能转换成列表,不能变成数组
            # list_temp = np.array((dict_temp.values()))
            Feature[i][:] = list_temp  # 把临时的节点系数值存放到特征矩阵中
        Feature_normalized = dataprocessing.MaxMinNormalized(
            Feature, mode=0, reverse=False)  # 归一化处理
        # 这里Feature是类似100×32的数据，那么应该按行归一化，也就是不同节点的值进行归一化,按列就是同一节点归一化了
        if tree:  # 只绘制最后1列数据
            for i in range(len(nodelist)):
                # plt.plot(nodelist[i])
                # plt.show()
                dataprocessing.plot(np.arange(0, len(nodelist[i]), 1), nodelist[i], titlename=[
                                    name + "  " + leafmode[i]], legendname=leafmode[i], Color=False)
        if plot:
            if mode == 0:  # 纵向分布 只是用于多组样本比对，观察是否具有普遍性
                # Feature = Feature.T # 转置 用于转换成dataframe格式
                Feature_temp = pd.DataFrame(
                    Feature_normalized,
                    columns=leafmode,
                    index=Signals.columns)  # 按节点名字给每列命名 , 输入信号的列名(如故障类型用于给行命名)
                Feature_temp['class'] = Signals.columns  # 创造新的一列作为分类标签
                for i in range(len(leafmode)):
                    sns.displot(
                        Feature_temp,
                        x=leafmode[i],
                        hue="class",
                        stat="density",
                        multiple="stack",
                        binwidth=5,
                        discrete=True)  # 得到同一故障类型不同样本的相同节点系数的分布
                    #sns.displot(Feature_temp, x=leafmode[i], hue="class", kind="kde", bw_adjust=.25,cut=0,fill=True,multiple="stack")
                    plt.show()
            if mode == 1:  # 横向分布 对应意义是能量矩
                '''绘制每个样本所有节点系数的分布'''
                Feature_temp = pd.DataFrame(
                    Feature_normalized,
                    columns=leafmode,
                    index=Signals.columns)
                Feature_temp = Feature_temp.T  # 节点数×样本数 32×100
                if which == 1:
                    for i in range(Feature_temp.shape[1]):  # 循环画出每1列的kde图
                        # 这里Signals.columns必须是字符串命名，否则数字命名列可能会出错
                        sns.displot(
                            Feature_temp,
                            x=Signals.columns[i],
                            bw_adjust=0.25,
                            kind="kde")
                        plt.show()
                if which == 0:
                    for i in range(Feature_temp.shape[1]):
                        sns.displot(
                            Feature_temp,
                            x=Signals.columns[i],
                            bins=20,
                            multiple="stack",
                            discrete=False)
                        plt.show()
                if which == 2:
                    pass
        return Feature_temp


class feature_predict():
    '''特征预测类'''

    def __init__(self, dataframe):
        self.dataframe = dataframe

    # 函数1 初始化参数
    def SetUp(self, target, trainsize=0.7):
        clf = setup(
            data=self.dataframe,
            target=target,
            train_size=trainsize,
            sampling=True,
            sample_estimator=None,
            silent=False,
            verbose=True,
            session_id=None,
            numeric_imputation='mean',
            categorical_imputation='constant',
            numeric_features=None,
            categorical_features=None,
            ignore_features=None,
            date_features=None,
            feature_selection=True,
            feature_selection_threshold=0.8)
        return clf

    # 函数2 比较各类模型的好坏
    def CompareModels(self, fold=10, round=4, sort='Accuracy', n_select=1):
        best = compare_models(
            fold=fold,
            round=round,
            sort=sort,
            n_select=n_select)
        return best

    # 函数3 创建模型
    def CreateModels(self, estimator, fold=10, round=4, cross_validation=True):
        model = create_model(
            estimator=estimator,
            ensemble=False,
            method=None,
            fold=fold,
            round=round,
            cross_validation=cross_validation,
            verbose=True,
            system=True)
        return model

    # 函数4 校准模型
    def CalibrationModels(self, model, fold=10, round=4):  # model是一已创建好的模型
        calibrated_model = calibrate_model(
            estimator=model,
            method='sigmoid',
            fold=fold,
            verbose=True,
            round=round)
        return calibrated_model

    # 函数5 预测模型
    def PredictModels(self, model, dataframe):
        '''model是已经窗创建好的模型，可以带入createmodel或者calibrationmodel得到的模型，dataframe是新数据'''
        pre = predict_model(
            estimator=model,
            platform=None,
            authentication=None)  # 返回模型的得分和预测标签
        # 预测其他数据的准备工作
        pre_final = finalize_model(model)
        Pre = predict_model(
            pre_final,
            data=dataframe,
            probability_threshold=None,
            platform=None,
            authentication=None,
            verbose=None)  # 新数据的预测标签和得分
        return pre_final, Pre  # 返回模型的得分和预测标签 / 和返回新数据预测的得分和标签

    # 函数6 保存模型
    def SaveModels(self, model, name):  # model是一已创建好的模型 或者 校准后的模型
        save_model(model, name)


# %%
if __name__ == '__main__':
    # %%
    # 调试程序1
    tcpClient = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 客户端
    tcpClient.connect(('192.168.1.198', 1600))
    self_tcp = tcp(tcpClient)  # 实例化类
    '''设置采集命令'''
    Settings = self_tcp.ADset()   # 成功时返回值 = 1
    '''启动采集命令'''
    Start = self_tcp.ADstart()   # 成功时返回值 = 1
    sample_time_start = datetime.datetime.now()
    print("数据采集开始", sample_time_start)
    sample_time_end = sample_time_start + datetime.timedelta(seconds=10)
    print("数据采集计划结束", sample_time_end)
    DATA_tuple = []  # 16进制元组
    # DATA_str = []   #16进制字符串
    DATA_signed10 = []  # 有符号十进制数
    while True:
        '''数据接收'''
        temp_data, temp1, temp2 = self_tcp.ADDataRead()  # 1440个字节
        '''数据处理'''
        dt, dict_10, dict_tmp = self_tcp.csv_tuple_str(temp_data)
        DATA_tuple.append(dict_tmp)
        # DATA_str.append(Dict_tmp)
        DATA_signed10.append(dict_10)
        time_now = datetime.datetime.now()
        if dt > sample_time_end:
            print("数据存储结束", time_now)
            break
        df_tuple = pd.DataFrame(DATA_tuple)
        # df_str = pd.DataFrame(DATA_str)
        df_signed_10 = pd.DataFrame(DATA_signed10)
        df_tuple.to_csv(
            r"C:/Users\chenbei\Desktop\钢\data_tuple.csv",
            index=False)  # 不保存标签
        # df_str.to_csv("C:/Users\chenbei\Desktop\钢\data_str.csv", index=False)
        df_signed_10.to_csv(
            r"C:/Users\chenbei\Desktop\钢\data_signed_10.csv",
            index=False)
    tcpClient.close()
    # %%
    # 调试程序2
    dataprocessing.ProgressBar(mode=True)  # 为True时只返回第一句
    starttime = time.perf_counter()
    data = pd.read_excel(r"C:/Users\chenbei\Desktop\钢\分合闸储能波形\分合闸储能波形\储能.xlsx")
    timenow = datetime.datetime.now()
    Index = pd.date_range(
        timenow,
        periods=len(data),
        freq='S')  # 可以选择D，Y，M，H，m，S
    data.index = Index
    del data['采样点']
    data_20 = data.resample('100S').sum()  # 100倍重采样
    T = data_20.values / abs(np.max(data_20.values))
    plt.plot(T[700:-1])
    plt.show()
    Temp = data_20.values.reshape(1, -1)[0]
    '''现在不需要实例化类'''
    # self_dataprocess = dataprocessing(Temp[700:-1])  # 实例化一个类，传递self参数，否则后续使用方法会造成缺少self参数
    # ①取出来的值是列，必须先转置②取出来的数是两层数组，只需要最外层③太多的0会导致不收敛，需要截取一段

    imf = dataprocessing.emd(Temp[700:-1])
    plt.plot(imf[0])
    plt.show()
    plt.plot(imf[1])
    plt.show()
    plt.plot(imf[2])
    plt.show()

    # 数据归一化 测试程序3
    x0 = np.array(np.random.randn(10, 5))
    norm_x0_col = dataprocessing.MaxMinNormalized(x0, 0, reverse=False)
    norm_x0_row = dataprocessing.MaxMinNormalized(x0, 1, reverse=False)
    reverse_col = dataprocessing.MaxMinNormalized(
        norm_x0_col, 0, reverse=True, col_max=x0.max(
            axis=0), col_min=x0.min(
            axis=0))  # 对按列归一化的数据反归一化需要原始数据的每列最值
    reverse_row = dataprocessing.MaxMinNormalized(
        norm_x0_row, 1, reverse=True, row_max=x0.max(
            axis=1), row_min=x0.min(
            axis=1))  # 对按行归一化的数据反归一化需要原始数据的每行最值

    # 故障预测 测试程序4
    diabetes = pd.read_csv(
        r'C:/Users\chenbei\Documents\python数据\pycaret-master\datasets\diabetes.csv')
    self_predict = feature_predict(diabetes)  # 实例化故障预测类
    clf = self_predict.SetUp(target='Class variable')
    best = self_predict.CompareModels()
    print(best)
    LR = self_predict.CreateModels(estimator='lr')  # 创建逻辑回归模型
    LR_calibrate = self_predict.CalibrationModels(LR)  # 校准模型
    pre_model, Pre = self_predict.PredictModels(
        model=LR, dataframe=diabetes)  # 返回模型的得分和预测标签 / 和返回新数据预测的得分和标签
    self_predict.SaveModels(pre_model, '最终模型')  # 保存模型
    dataprocessing.ProgressBar(starttime)

    # 测试颜色/sns.pairplot 测试程序5
    diabetes = pd.read_csv(
        r'C:/Users\chenbei\Documents\python数据\pycaret-master\datasets\diabetes.csv')
    # dataprocessing.showcolor()
    #dataprocessing.ClassificationDrawing(diabetes,'Number of times pregnant','Age (years)','Class variable')
    dataprocessing.ClassificationDrawing(
        diabetes,
        'Number of times pregnant',
        'Age (years)',
        'Class variable',
        mode='plot')
    dataprocessing.RelationChart(
        diabetes,
        vars=[
            "Number of times pregnant",
            "Age (years)"],
        hue='Class variable',
        palette='husl',
        markers=[
            "o",
            "D"])

    # 测试五点三次平滑法 测试程序6
    now = datetime.datetime.now()
    ecg = pywt.data.ecg()
    ecg_t = dataprocessing.Mean_3_5(ecg, 3)
    now1 = datetime.datetime.now()
    print(now1 - now)
    #ecg = pd.Series(ecg)
    #ecg.to_csv("C:/Users\chenbei\Desktop\钢\ecg.csv", index=False)
    #index, values = dataprocessing.FindExtremum(ecg_t)

    # 测试寻找极值函数 测试程序7
    x = np.array([1,2,3,3.1,5,7,8,9,9.1,9.2,9.4,9.6,9.8,10,10.2,10.3,10.8,
                  11,10.8,10.6,10.1,10,9.5,9.3,9,6,5.8,5.7,5.6,5.5,5.3,5.1,8,8.8,9,9.5,10,10.4,
                  11,15,16,18,17.5,17,16,14.4,14,13.5,13,12,9.4,9,8.8,7.5,7])
    index, values = dataprocessing.FindExtremum(x)
    feature = dataprocessing.Statistics(x)
# %%
    # 测试小波能量矩/横向分布核密度估计图 测试程序8
    D = pd.read_csv(r"C:/Users\chenbei\Desktop\钢\D.csv")
    #k = feature_extract.WaveletAlternation(D.values)
    #Feature = feature_extract.WaveletAlternationEnhance(D, 'sym4', 4, tree=True,plot=False)
    Feature = feature_extract.WaveletAlternationEnhance(
        D, 'db3', 3, tree=False, plot=True, which=0)
    # %%
    # 饼图绘制 测试程序9
    dataprocessing.PythonPie()
'''
①my name is chenbei , I am a second-year student of the school of Electrical Engineeering of the BeiJing JiaoTong University;
②In terms of learning , my research direction is about the fault diagnoisis of low-voltage circuit breakers ;
Relatively speaking ,I am more familiar with your company's products;
③In terms of skills , I am better at C and Python programming , and the use of office softwares and matlab;
④Last , my basketball is also good , I have won the team first as a main player many times during colleage . Thank you!
'''
'''
括号运算符 : '()', '{}' ,'[] '
其它  : '->' 指向结构体成员运算符 , '.' 结构体成员运算符
地址运算符 : '&'
长度运算符 : 'sizeof'
逻辑运算符 : '&&' , '||' , '!' 与或非
算术运算符 : '+','-','/','*','+','-'  加减乘除 , 单目运算符 ±  同优先级别 自左向右
自增自减运算符 : '++' , '-- '
关系运算符 : '>', '<' , '>=' , '<=' , '==' , '!='
位逻辑运算符 : '&' , '|' , '^' , '~' 与或/异或/取反   // 只能用于整型变量
赋值运算符 : '=' ,'+=' , '*=' , '-=' , '/=' ,'^=','<<=','>>=','&=','|='
逗号运算符 : ',' 表达式1，表达式2,表达式3,... (级别最低)
移位运算符 : '<<' , '>>'
三目运算符 : '? : '
'''
