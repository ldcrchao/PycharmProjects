#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
'''统计图常用API : scatterplot + relplot + lineplot'''
'''
1、relplot(data,x,y,hue,style...) 
parms data : 数据dataframe格式
parms x , y: 指定绘制的两列关系 ，type 字符串 ：列名称
parms hue : type = 字符串，指定按照某列(类) 对数据划分 ，采用不同的色调
parms size : 按类改变标记大小 ，一般size 和 hue 选用一种即可 ，视觉上推荐采用颜色区分 有更好的效果
parms sizes : 改变标记大小的范围选取  元组格式 = (15,200)
parms style : type = 字符串 除了颜色不同，不同类还将采样不同的标记 ，style 和 hue 也可以使用不同的列，这样会出现4种情况  
parms palette : "ch:r=-0.5,l=0.75" :type = 字符串 使用多维数据集heubehelix_palette（）的字符串接口来自定义顺序调色板 改变 r , l 的值即可
parms kind  :  "line" 、 "scatter" 可以使用lineplot() 或者kind强制为line  relplot =scatterplot + lineplot
parms sort : 默认 sort 的每个点按横坐标由小到大排序 ， sort = False 时出现 无序折线图 
parms ci : 禁用None 置信区间 , seaborn 默认使用置信区间95%的绘图 ,  或者 “sd” 使用 绘制偏差而不是置信区间 ，可以节省时间
parms row 、 col : row ，col 字符串 ，可以表示多重关系 
parms estimator : None 可以关闭一个点具有的多重值 ，以防止产生奇怪的效果
parms height=3, aspect=0.75, linewidth=2.5  : 高度 、宽度、 线宽
parms col_wrap=5  : 多重关系时指定 按5列绘图
2、scatterplot() 参数与relplot类似
3、lineplot() 
'''

#tips = sns.load_dataset("tips") #dataframe格式
#sns.relplot(x="total_bill", y="tip", hue="smoker",row="sex",col="time", data=tips ,height=2 ,aspect=2);
#sns.scatterplot(x="total_bill",hue="smoker",style="smoker", y="tip", data=tips)  # 效果上区别不大，图例出现的位置不一样
#plt.show()
#sns.lineplot(x="total_bill",hue="smoker",style="smoker", y="tip", data=tips)  # 折线图参数也是类似的
#plt.show()
#sns.relplot(x="total_bill",hue="smoker",style="smoker", y="tip", data=tips,kind='line' , ci = 'sd') # kind 转换后没有任何区别
#plt.show()

#sns.relplot(x="total_bill",hue="smoker",style="smoker", y="tip", data=tips)
#sns.relplot(x="total_bill", y="tip", hue="smoker", style="time", data=tips); # 不吃晚饭，不吃午饭 的吸烟者 和 不吸烟者
#sns.relplot(x="total_bill", y="tip", hue="size", data=tips);# smoker只有2种选择，size多种，就会以深浅色出现
#sns.relplot(x="total_bill", y="tip", hue="size", palette="ch:r=-1.5,l=0.75", data=tips); # 调色板 ，改变 r、l值
#sns.relplot(x="total_bill", y="tip",size="size",sizes=(15,200), data=tips); #不使用颜色进行区分，也可以使用大小进行区分，或者同时使用(但是区分度不高，最好只选用一种)

#df = pd.DataFrame(np.random.randn(500, 2).cumsum(axis=0), columns=["x", "y"])
#sns.relplot(x="x", y="y", sort=False, kind="line", data=df)
#sns.relplot(x="x", y="y", sort=True, kind="line", data=df)

'''分布图常用API : 
轴级 : histplot + kdeplot + ecdfplot + rugplot
图级 : displot + pairplot
1、displot
data : dataframe格式
x , y : x单变量分布 ,y 双变量分布
*binwidth : double 步长 , 可以让直方图在更细致的范围统计
bins : int 不指定步长 , 指定柱的个数 更有意义
*discrete : bool , 定条形图的中心是横坐标值 , 用于默认 bin 宽度太小，分布中有较大间隙的情况
*shrink : double=0.8 ， 逻辑可视化分类变量的分布 , 按类 收缩 条形图 ，使之效果是离散的 
hue : string 按类绘图
*element : string , "step" 将同一类型的柱子之间的界限去除 , 变为步进图
multiple : string , "stack" 堆叠直方图，而不是分层每个条形 ； "dodge" "闪"条，只有当分类变量具有少量级别时才能正常工作
col : string , 按照某列水平堆叠不同类的相同关系图
*stat : 归一化直方图 "density" , "probability" 密度归一化缩放标尺，以使它们的面积总和为1
common_norm : False ,bool ,独立规范化每个子集
kind : string , 直方图旨在通过装箱和计数观测值来近似生成数据的基础概率密度函数"hist"
KDE图不使用离散条柱，而是使用高斯内核平滑观测，从而生成连续密度估计值"kde"
经验累积分布函数绘制一条单调增加的曲线，通过每个数据点，使曲线的高度反映具有较小值的观测值的比例 "ecdf" , ECDF图的主要缺点是，它表示分布的形状不如直方图或密度曲线直观
kde参数选项：
bw_adjust : double= 0.25 与直方图中的 bin 大小非常不同，KDE 准确表示数据的能力取决于平滑带宽的选择
fill : bool = True 改变透明度值,各个密度更易于解析 ,stack会导致第2类颜色覆盖掉重复的第1类
*cut : int = 0 ,剪除极端数据点
*cbar : bool = True 双变量热图 使用颜色深浅解释热图
*thresh 、 level : double 双变量密度等高线 ，不常用的参数
*log_scale : bool = (True,False)指定两个变量离散或者连续
*rug: bool = True 显示边际分布的不太显眼的方法使用"rug"绘图
2、jointplot 关节图(默认散点图+直方图)《-》(双变量分布+边际分布)
3、rugplot 地毯图
4、pairplot 矩阵图
'''
penguins = sns.load_dataset("penguins")
#sns.jointplot(data=penguins, x="bill_length_mm", y="bill_depth_mm") # 关节图  默认散点图+直方图
#sns.jointplot(data=penguins,x="bill_length_mm", y="bill_depth_mm", hue="species",kind="kde") #配合hue , 使用核密度估计代替直方图,双变量代替散点图

#sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm",kind="kde", rug=True)#显示边际分布的不太显眼的方法使用"rug"绘图

#sns.rugplot(data=penguins, x="bill_length_mm", y="bill_depth_mm")#轴级地毯图（）功能可用于在任何其他种类的绘图的侧面添加地毯

#sns.displot(penguins, x="flipper_length_mm",binwidth=3) #单变量直方图
#sns.displot(penguins, x="flipper_length_mm",bins=20) #指定柱的个数

#sns.displot(penguins, x="flipper_length_mm", kind="ecdf")#经验累积分布图
#sns.displot(penguins, x="flipper_length_mm", hue="species", kind="ecdf") #没有要考虑的 bin 大小或平滑参数,曲线是单调增加的，因此非常适合比较多个分布
#plt.show()

#sns.displot(penguins, x="flipper_length_mm", kind="kde" ) # 内核密度估计图
#sns.displot(penguins, x="flipper_length_mm", kind="kde", bw_adjust=.25) #选择平滑带宽
#sns.displot(penguins, x="flipper_length_mm", hue="species", kind="kde",bw_adjust=.25) #配合其他变量hue
sns.displot(penguins, x="flipper_length_mm", hue="species", kind="kde", multiple="stack") # multiple ,密度曲线下方包络区域会填充颜色
sns.displot(penguins, x="flipper_length_mm", hue="species", kind="kde", fill=True) # 改变透明度值 , 可以看到不被覆盖的曲线

#sns.displot(penguins, x="flipper_length_mm", hue="species",element="step")#步进图
#sns.displot(penguins, x="flipper_length_mm", hue="species", multiple="stack")#堆叠直方图
#sns.displot(penguins, x="flipper_length_mm", hue="sex", multiple="dodge") # 闪条
#sns.displot(penguins, x="flipper_length_mm", col="sex") # 按col指定的sex男女两类堆叠

#sns.displot(penguins, x="flipper_length_mm", hue="species") # 不归一化
#sns.displot(penguins, x="flipper_length_mm", hue="species", stat="density") #归一化
#sns.displot(penguins, x="flipper_length_mm", hue="species", stat="probability") # 密度归一化缩放标尺
#sns.displot(penguins, x="flipper_length_mm", hue="species", stat="density", common_norm=False) #独立规范化每个子集

tips = sns.load_dataset("tips") # 默认的bin太小 导致柱子之间存在间隙
#sns.displot(tips, x="size")
#sns.displot(tips, x="size", bins=[1, 2, 3, 4, 5, 6, 7])#1、可以直接传递列表进行中断
#sns.displot(tips, x="size", discrete=True) # 2、也可以设置discrete来指定条形图的中心是横坐标值
#sns.displot(tips, x="day", shrink=0.8) # 按类收缩条形图

#sns.displot(tips, x="total_bill", kind="kde")
#sns.displot(tips, x="total_bill", kind="kde", cut=0) # 剪除极端数据点

'''可视化双变量分布'''
#sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm")
#sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", kind="kde") #类似于等高线
#sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", hue="species",kind="kde") # 配合hue的等高线
#sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", binwidth=(2, .5)) #平滑带宽需要一对值
#sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", binwidth=(2, .5), cbar=True) #cbar可以映射颜色解释热图

'''双变量密度等高线'''
#sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", kind="kde", thresh=.2, levels=4) # thresh \ levels
#sns.displot(penguins, x="bill_length_mm", y="bill_depth_mm", kind="kde", levels=[.01, .05, .1, .8])#参数还接受值列表，以进行更多控制
#sns.displot(diamonds, x="price", y="clarity", log_scale=(True, False))#双变量直方图允许一个或两个变量是离散的
plt.show()

