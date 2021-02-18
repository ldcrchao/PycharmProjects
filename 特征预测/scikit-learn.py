#%%
#线性回归
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  # 线性模型下的线性回归
model = LinearRegression() #创建一个线性回归实例
X = np.array([[6],[8],[10],[14],[18]]).reshape(-1,1) # 1列 ，披萨的直径 二维数组，例如7还可以包含7.1,7.2等
y = np.array([7,9,13,17.5,18]) # 披萨价格
model.fit(X,y) # 训练数据
x0 = np.array([[8],[9],[11],[16],[12]])
pre = model.predict(x0)   # 预测价格
real = np.array([11,8.5,15,18,11]) #真实价格
print(pre)
meanloss = np.mean((model.predict(X)-y)**2) # 代价函数公式 : loss = Σ(yi-f(xi))^2 预测自身的值与实际值的差平方和
variance = ((X-X.mean())**2).sum()  / (X.shape[0]-1) #计算方差 : var = Σ(x-xmean)^2 / (n-1) n为数据长度 贝塞尔校正
'''协方差衡量两个变量如何一同变化，正相关或负相关，如果为0表示两者没有关系 ； 方差衡量数据到均值的偏离程度'''
cov = np.multiply((X-X.mean()).transpose(),y-y.mean()).sum() / (X.shape[0]-1)#协方差 : cov = Σ(y-ymean)(x-xmean)/(n-1)
'''最佳拟合y=a+bx，其中b=cov(x,y)/var(x),a = ymean - b * xmean ，其中x_和y_为质心坐标，是一个模型必经点'''
b = cov / variance
a = y.mean()-b*X.mean()
'''协方差和方差是衡量模型本身的拟合能力，现在还需要评价预测能力 ，使用参数为R方，其等于皮尔森积差相关系数的平方(PPMCC)'''
r_squared = model.score(x0,real) # 预测数据和真实价格之间的关系
#%% k_近邻算法(KNN)
'''1、knn练习'''
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
x_train = np.array([[158,64],[170,86],[203,95],[183,84],[191,80],[155,49],[163,59],[180,67],
                    [158,54],[170,67]])
y_train = ['male','male','male','male','male','female','female','female','female','female']
plt.figure()
for i , x in enumerate(x_train ):
    plt.scatter(x[0],x[1],c='k',marker='x' if y_train[i] =='male' else 'D')
plt.grid(True)
plt.xlabel('身高')
plt.ylabel ('体重')
plt.show()
x0 = np.array([[155,70]])
distance =np.sqrt(np.sum((x_train -x0)**2,axis=1)) # 计算给定要预测的x0与其它训练实例的距离值
knn_max_index = distance.argsort()[:3] # 返回从小到大的索引值，这里取最小的3个
knn_max_genders = np.take(y_train ,knn_max_index ) #根据提供的索引寻找对应的元素
b = Counter(np.take(y_train,distance.argsort()[:3]))
print(b.most_common()[1] ) #返回n个最常见的元素及其个数，以元组形式返回，如果n没有给定或者超过种类数则返回全部
b.most_common(1)[0][0] #索引 ： 第1个0表示返回第一组元组对，第2个0返回元组对中的key值
'''调用scikit-learn的代码'''
from sklearn.preprocessing import LabelBinarizer  #打标签的包
from sklearn.neighbors import KNeighborsClassifier
lb = LabelBinarizer () # 实例化1个类
y_train_binarizer = lb.fit_transform(y_train ) #返回的是列向量
K = 3 #返回最近的邻居数
clf = KNeighborsClassifier(n_neighbors= K)
clf.fit(x_train ,y_train_binarizer.reshape(-1)) # 训练函数要求 训练数据是样本数×维数，标签是行向量
pre = clf.predict(x0.reshape(1,-1)) # x0 其实应该理解为 元素为 元素对 的数组，所以再预测我们其实需要预测的其中的某一个元素对 ,x0本来就是行形式的向量
x_test = np.array ([[168,65],[180,96],[160,52],[169,67]])
y_test=['male','male','female','female']
y_test_binarizer = lb.fit_transform(y_test )
pre_test = clf.predict(x_test)
pre_test = lb.inverse_transform(pre_test ) # 测试集的预测
from sklearn.metrics import accuracy_score  # 计算正向类准确率的包
print(accuracy_score(y_test,pre_test )) # 预测男性为男性，且女性为女性的比例
from sklearn.metrics import precision_score
print(precision_score(y_test_binarizer ,clf.predict(x_test) )) # 精准率 ,预测为男性且确实为男性的比例
from sklearn.metrics import recall_score
print(recall_score(y_test_binarizer ,clf.predict(x_test) )) # 召回率 ,实际为男性且预测为男性的比例
from sklearn.metrics import f1_score  # F1度量，用于求解精准率和召回率的调和平均值
print(f1_score(y_test_binarizer ,clf.predict(x_test))) # 精准率和召回率的平均值是F1度量的上界
from sklearn.metrics import matthews_corrcoef # MCC 马修斯相关系数
print(matthews_corrcoef(y_test_binarizer ,clf.predict(x_test))) # 完美的分类器得分为1，随机进行预测的分类器得分0，完全预测错误的得分-1
from sklearn.metrics import classification_report # 一次性生成准确率、精准率和召回率的包
print(classification_report(y_test_binarizer ,clf.predict(x_test),target_names=['male'],labels=[1]))
#label:报告中要包含的标签索引的可选列表 #target_names:与标签匹配的可选显示名称（相同顺序）
#%%
'''knn回归'''
'''通过身高和年龄预测体重，上述问题是通过身高和体重预测性别'''
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_absolute_error , mean_squared_error ,r2_score #引入均方根误差和绝对误差以及R2得分
X_train = np.array([[158,1],[170,1],[183,1],[191,1],[155,0],[163,0],[180,0],[158,0],[170,0]])
X_test = np.array ([[168,1],[180,1],[160,0],[169,0]])
y_train = [64,86,84,80,49,59,67,54,67]
y_test = [65,96,52,67]
K  = 3
clf = KNeighborsClassifier (n_neighbors= K)
clf.fit(X_train,y_train)
pre = clf.predict(X_test )
#np.array_equal (pre,y_test )
c = (pre==y_test)
#c.any() # 有1个相等即返回True
#c.all() # 都相等才返回True
print(c)
print("均方根误差为：",mean_squared_error(y_test ,pre)) #MSE
print("均方偏差为：",mean_absolute_error(y_test ,pre)) #MAE
print("R2得分为：",r2_score(y_test ,pre))
'''特征缩放，性能变好'''
from sklearn.preprocessing import StandardScaler #归一化
ss = StandardScaler ()
X_train_scaled = ss.fit_transform(X_train)
#fit_transform()的功能就是对数据先进行拟合处理，然后再将其进行标准化
#可以看做是fit和transform的结合，如果训练阶段使用fit_transform，则在测试阶段只需要对测试样本进行transform
X_test_scaled = ss.transform(X_test ) #对数据进行标准化，与fit_transform(X[,y])的结果是一样的
X_test_scaled1 = ss.fit_transform(X_test )
clf.fit(X_train_scaled,y_train)
Pre = clf.predict(X_test_scaled)
Pre1 = clf.predict(X_test_scaled1 )
print("特征缩放后的均方根误差为：",mean_squared_error(y_test ,Pre)) #MSE
print("特征缩放后的均方偏差为：",mean_absolute_error(y_test ,Pre)) #MAE
print("特征缩放后的R2得分为：",r2_score(y_test ,Pre))
