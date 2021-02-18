#%%
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy import interpolate
import pywt
from matplotlib.pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] # 显示中文的命令
mpl.rcParams['axes.unicode_minus'] = False  #  显示负号的命令
#经验模态函数EMD

#函数1：判定当前时间序列是否是单调序列
def mon(x):
    max_peaks=sig.argrelextrema(x,np.greater)[0]#极大值点横坐标，np.greater实现的是">"的功能
    min_peaks=sig.argrelextrema(x,np.less)[0]#极小值点横坐标
    num=len(max_peaks)+len(min_peaks)#极值点个数
    if num >0:#不是单调序列
        return False
    else:
        return  True
#函数2：寻找当前时间序列的极值点横坐标
def findpeaks(x):
    return sig.argrelextrema(x,np.greater)[0]
#函数3：判断是否为IMF序列
def  imfyn(x):
    N = np.size(x)
    pzero = np.sum(x[0:N-2] * x[1:N-1] < 0)#过零点个数
    psum = np.size(findpeaks(x)) + np.size(findpeaks(-x))#极值点个数
    if abs(pzero - psum) > 1:
        return False
    else:
        return True

#函数4：获取样条曲线
def getspline(x):
    N = np.size(x)
    peaks = findpeaks(x)
    peaks = np.concatenate(([0],peaks))
    peaks = np.concatenate((peaks,[N - 1]))
    if (len(peaks) <=3):
        t = interpolate.splrep(peaks,y=x[peaks] , w=None , xb=None , xe=None ,k=len(peaks) -1)
        return interpolate.splev(np.arange(N),t)
    t = interpolate.splrep(peaks, y=x[peaks])
    return interpolate.splev(np.arange(N),t)
#函数5：CBemd
def emd(x):
    imf= []
    while not mon(x):
        x1 = x
        sd = np.inf
        while sd >0.1 or (not imfyn(x1)):
            s1 = getspline(x1)
            s2 = -getspline(-1*x1)
            x2 = x1 - (s1 + s2)/2
            sd = np.sum((x1-x2)**2) / np.sum(x1 ** 2)
            x1 = x2
        imf.append(x1)
        x = x -x1
    imf.append(x)
    return imf
# 函数6：模拟噪声函数wgn
def wgn(x,snr):
    snr = 10**(snr/10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x) )* np.sqrt(npower)
# 函数7：matlab仿写的软阈值去噪函数
def Thr(x):
    len1 = len(x)
    w = sorted(x) #从小到大排序
    if len1 % 2 == 1:
        v = w[int((len1+1)/2)-1] # 如len1=5，计算结果为3 ，但是3不是中间位置还需要减1
    else:
        v = (w[int(len1/2)-1] + w[int(len1/2)]) / 2 #是 偶数就 取中间两个值的平均值
    sigmal = abs(v) / 0.6745
    valve = sigmal * ( ( 2 * ( math.log(len1, math.e)  ) ) ** (0.5) )
    return valve
# 函数8：python小波去噪函数
def  quzao(data):
    mode = 'db38'
    w = pywt.Wavelet(mode)
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)#找到数据可以分解的最大水平
    #maxlev = 3 #一般可以取3
    print(f"matlab小波去噪函数 - 原始数据在{mode}小波基下的最大分解水平为："  + str (maxlev) ) # 这是可以分解的最大水平，但是不推荐，本信号最大可分解6层，但是3 最好
    coeffs = pywt.wavedec(data , w ,level = int(maxlev / 2 ) )# 高频细节系数 ，#返回CAn,CDn,CDn-1,CDn-2,....CD1
    for j in range(1 , len(coeffs)) : # 外循环，不同的细节系数轮流去噪 ,从1开始，因为第1个是近似系数
        valve = Thr(coeffs[j]) #当前的细节系数 计算阈值
        temp = coeffs[j] #把当前细节系数给副本temp ，整行赋值
        for i in range(int(len(temp))):#内循环 ，对当前细节系数每一个元素进行处理 ， 循环次数取决于 当前细节系数的长度
            if (abs(temp[int(i)]) <= valve):#绝对值小于等于阈值的都取0
                temp[int(i)] = 0
            else:
                if (temp[int(i)] > valve ):
                    temp[int(i)] = temp[int(i)-1] - valve  #绝对值大于阈值的都减去阈值
                else:
                    temp[int(i)] = temp[int(i) - 1] + valve #其他情况 ，比如很大的负数，加上阈值
        coeffs[j] = temp #可以这样赋值，把当前去噪后的细节系数替换掉 存放细节系数的coeffs
    datarec = pywt.waverec(coeffs , w) #外循环结束 ，  所有替换过的 coeffs 再做逆重构
    return datarec
#函数9：db8 小波去噪函数
def quzao_1(x):
    w = pywt.Wavelet('db8')
    maxlev = pywt.dwt_max_level(len(x), w.dec_len )
    print("python小波去噪函数 - 原始数据在db8小波基的最大分解水平为： " + str(maxlev))
    threshold = 0.04
    coeffs = pywt.wavedec(x, 'db8',level = maxlev )
    for i in range(1, len(coeffs)): #只对细节系数处理 ，也就是高频分量去噪
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))
    datarec = pywt.waverec(coeffs,'db8') #将信号小波重构
    return  datarec
#函数10：自定义的绘图函数Plot
def Plot(x,y,legendname,titlename,xlabelname,ylabelname):
    '''设置图片大小，分辨率，[图像编号]，背景色，是否显示边框 ， 边框颜色'''
    plt.figure(figsize=(8,4) , dpi= 144, facecolor= 'w' , frameon= True  ,  edgecolor= 'b')

    '''设置图例文字，线颜色，线型，线宽，线风格，标记类型，标记尺寸，标记颜色，透明度'''
    plt.plot(x,y,label=legendname, color='r' , linewidth = 0.6, linestyle =  '--' , marker = 'o',
             markersize = '2' , markerfacecolor = 'w' , alpha = 0.5)

    '''设置图例的文字，位置 ， 文字大小， 多图例按列数显示(默认按行)， 图例标记为原尺寸的多少倍大小 ， 是否添加阴影 ， 图里圆角方角，是否带边框， 边框透明度'''
    plt.legend((legendname,), loc= 'upper right' , fontsize = 18,ncol = 1, markerscale=1, shadow=False , fancybox=True,  frameon = True , framealpha=0.5)

    '''设置网格线是否出现、显示哪个轴的网格线 ，网格线颜色， 风格、 宽度、 设置次刻度线（default = major）'''
    #plt.grid(b=None, axis= 'x', color ='k', linestyle = '-.',linewidth = 5 , which= 'major')

    '''设置坐标轴范围'''
    ymax = max(y)
    ymin = min(y)
    xmax = max(x)
    xmin = min(x)
    plt.ylim(ymin , ymax)
    plt.xlim(xmin , xmax)

    '''设置轴标签是否旋转，文字垂直和水平位置'''
    plt.xlabel(xlabelname,fontsize=18 ,rotation = None , verticalalignment = 'top' )
    plt.ylabel(ylabelname,fontsize=18 ,rotation = 90 , verticalalignment = 'bottom' )

    '''设置轴刻度'''
    plt.yticks(())
    plt.xticks(())

    '''设置哪个轴的属性、次刻度线设置， 刻度线显示方式(绘图区内测、外侧、同时显示)，刻度线宽度，刻度线颜色、刻度线标签颜色(任命/日期等)
    刻度线与标签距离、刻度线标签大小、刻度线是否显示(default=bottom ,left)、刻度线标签是否显示(default = bottom/left)'''
    plt.tick_params (axis='both',which='major' , direction = 'in' , width=1,length=3,color='k',labelcolor='k', pad=1,
                     labelsize = 15, bottom = True , right =True , top = False, labeltop=False,labelbottom = True ,
                                    labelleft = True , labelright = False)

    '''方框外形、背景色、 透明度、方框粗细、方框到文字的距离'''
    bb = dict(boxstyle ='round',fc='w', ec = 'm',alpha=0.8, lw=10, pad= 0.6)

    '''设置标题文字大小、标题大小、标题正常/斜体/倾斜、垂直位置、水平位置、透明度、标题背景色、是否旋转、标题边框有关设置（字典格式）'''
    plt.title (label=titlename , fontsize =24, fontweight='normal', fontstyle='italic',verticalaligment='bottom',
                alpha=0.8 , backgroundcolor = 'w', rotation = None )#bbox = bb

    plt.show()
#主程序main################################################
# 1、定义原始数据
fs=25000 #采样频率要大于 信号频率 的两倍
f0=2000
f1=6000
t = np.arange(0,1,1/fs)
#x = 0.6 * (1+np.sin(2 * np.pi * f0 * t)) * np.sin(2 * np.pi * f1 * t)
x = 4 + 6*np.sin(2000*np.pi*t) + 8*np.sin(6000*np.pi*t) + 10*np.sin(10000*np.pi*t)
#2、添加白噪声
x = x + wgn(x,3)
#3、使用函数9 db8小波基去噪
xrec = quzao_1(x)
ymax = max(xrec)
ymin = max(xrec)
#4、去噪前后图像比对，因为子图缘故不可以调用Plot函数
plt.figure(figsize= (8,4))
plt.subplot(2,1,1)
plt.plot(t,x,c='cornflowerblue',linestyle='-',linewidth=0.6)
plt.xlabel('时间 (s)')
plt.ylabel('幅值 (m)')
plt.xlim(min(t),max(t))
plt.ylim(min(x)+1,max(x)+1)
plt.legend(('source_signal',),loc='upper right',fontsize=10,frameon=True )
plt.title('source',fontsize=20)
plt.xticks(())
plt.yticks(())

plt.subplot(2,1,2)
plt.plot(t,xrec,c='lightslategrey',linestyle='-',linewidth=0.6)
plt.xlabel('时间 (s)')
plt.ylabel('幅值 (m)')
plt.xlim(min(t),max(t))
plt.ylim(min(xrec)+1,max(xrec)+1)
plt.legend(('denoised_signal',),loc='upper right',fontsize=10,frameon=True )
plt.title('denoised',fontsize=20)
plt.xticks(())
plt.yticks(())

plt.tight_layout ()#自动调整子图间距，使之填充整个图像区域
plt.show()
#5、emd分解
imf = emd(xrec)
#6、imf分量绘制 前4个imf分量
fig , ax = plt.subplots (2,2)
plt.subplots_adjust (wspace= 0.4,hspace=0.4)#调整子图大小
axs = ax.flatten() #面向对象，将子图句柄展开，其中每个元素代表1个子图
'''子图分别绘制'''
'''imf1'''
axs[0].plot(t , imf[0] , color='lightseagreen' , linewidth = 0.6 ,linestyle = '-')
axs[0].set_title('imf1', fontsize=15)
axs[0].legend(('imf1',),loc = 'upper right',fontsize=10,frameon =True)
axs[0].set_xticks(())
axs[0].set_yticks(())
'''imf2'''
axs[1].plot(t , imf[1] , color='deepskyblue' , linewidth = 0.6 ,linestyle = '-')
axs[1].set_title('imf2', fontsize=15)
axs[1].legend(('imf2',),loc = 'upper right',fontsize=10,frameon =True)
axs[1].set_xticks(())
axs[1].set_yticks(())
'''imf3'''
axs[2].plot(t , imf[2] , color='blueviolet' , linewidth = 0.6 ,linestyle = '-')
axs[2].set_title('imf2', fontsize=15)
axs[2].legend(('imf2',),loc = 'upper right',fontsize=10,frameon =True)
axs[2].set_xticks(())
axs[2].set_yticks(())
'''imf4'''
axs[3].plot(t , imf[3] , color='dimgrey' , linewidth = 0.6 ,linestyle = '-')
axs[3].set_title('imf3', fontsize=15)
axs[3].legend(('imf3',),loc = 'upper right',fontsize=10,frameon =True)
axs[3].set_xticks(())
axs[3].set_yticks(())

plt.show()
# 7、傅里叶变换
from scipy.fftpack import fft , ifft
fftx = fft(xrec)
mol = np.abs(fftx)
t_mol = mol[range(int(fs/2))]/(fs/2) #频域纵坐标到时域纵坐标的转换
freq = ( ( t[ range(int(fs/2 ) ) ] + 1/fs) * fs -1) #时域横坐标到频域横坐标的转换，并取一半
angle = np.angle(fftx) #复数模 和 相位
'''绘制原始信号的频谱图'''
plt.figure(dpi=144,figsize= (8,4))
plt.plot(freq , t_mol ,c = 'blueviolet' , linewidth=0.6 , linestyle = '-')
plt.title('去噪信号的频谱图', fontsize =20)
plt.legend(('频谱图',),loc = 'upper right' , fontsize=15 , frameon = True)
plt.show()
'''循环绘制4个imf分量的频谱图'''
fig1  = plt.figure(dpi=144,figsize= (8,6))
title = ('imf1','imf2','imf3','imf4')
for i in range(4):
    fftimf = fft(imf[i])
    molimf = np.abs(fftimf)
    angleimf = np.angle(fftimf)
    half_molimf = molimf[range(int(fs/2))] / (fs/2)
    freq = ((t[range(int(fs/2))] + 1/fs) * fs -1)
    Q=fig1.add_subplot(2,2,i+1)
    Q.plot(freq , half_molimf ,c = 'blueviolet' , linestyle='-' , linewidth = 0.6)
    Q.set_title(title[i],fontsize=20)
    Q.legend((title[i],),loc='upper right',fontsize=15 , frameon=True )
plt.tight_layout ()
plt.show()

#8、主成分分析PCA
from sklearn.decomposition import PCA
data_pca = np.array(imf) #转换为数组格式，原本是列表格式，imf是行向量
pca = PCA(n_components= 0.90 , copy =True ,whiten = False ) #找到占比95%以上的imf分量
pca.fit(data_pca )
pca_tran = pca.transform(data_pca )
print('主要的imf分量个数为：'+ str(pca.n_components_)) #返回被保留的主成分个数
print('各imf分量占比依次为：'+str(pca.explained_variance_ratio_))# 各imf分量的占比
#print(pca.explained_variance_ ) #找到特征向量的方差，也就是特征值
#print(pca.components_ ) #返回被保留的主成分值，也可以直接查看pca



























