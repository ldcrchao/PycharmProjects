#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : PCA_doc.py
@Author : chenbei
@Date : 2020/12/17 11:29
'''
# 参数1 : n_components 主成分降维的个数，如果没设置就会选择维数和样本数最小的那个减1
# n_components='mle'的条件是svd_solver = 'full' 且样本数大于维数 ; 一般来说在(0,1)之间即可
# 将自动选取主成分个数n，使得满足所要求的方差百分比
# n_components=2 代表返回前2个主成分
# 0 < n_components < 1代表满足最低的主成分方差累计贡献率
# n_components=0.98，指返回满足主成分方差累计贡献率达到98%的主成分
# n_components=None，返回所有主成分
# 如果svd_solver =='arpack'，则组件数量必须严格小于n_features和n_samples的最小值
# 参数2 : copy 默认 True , 一般原始训练数据不希望改变, 复制一份数据基础上降维
# 参数3 : whiten 如果为True（默认情况下为False），则将components_向量乘以n_samples的平方根，然后除以奇异值，以确保不相关的输出具有单位分量方差
# 目的就是降低输入数据的冗余性，使得经过白化处理的输入数据具有如下性质：(i)特征之间相关性较低；(ii)所有特征具有相同的方差。
# 参数4 : svd_solver , 可选{'auto', 'full', 'arpack', 'randomized'} 定奇异值分解 SVD 的方法
# 默认采用auto, auto是根据n_components 和 样本数进行自动选择的
# svd_solver=auto：PCA 类自动选择下述三种算法权衡。
# svd_solver=‘full’:传统意义上的 SVD，使用了 scipy 库对应的实现。
# svd_solver=‘arpack’:直接使用 scipy 库的 sparse SVD 实现，和 randomized 的适用场景类似。
# svd_solver=‘randomized’:适用于数据量大，数据维度多同时主成分数目比例又较低的 PCA 降维

# 属性
# 1. components_：返回最大方差的主成分。
# 2. explained_variance_：它代表降维后的各主成分的方差值。方差值越大，则说明越是重要的主成分。
# 3. explained_variance_ratio_：它代表降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分。（主成分方差贡献率）
# 4. singular_values_：返回所被选主成分的奇异值。
# 实现降维的过程中，有两个方法，一种是用特征值分解，另一种用奇异值分解，前者限制比较多，需要矩阵是方阵，而后者可以是任意矩阵，而且计算量比前者少，所以说一般实现PCA都是用奇异值分解的方式。
# 5. mean_：每个特征的经验平均值，由训练集估计。
# 6. n_features_：训练数据中的特征数。
# 7. n_samples_：训练数据中的样本数量。
# 8. noise_variance_：噪声协方差

# 方法
# 1. fit(self, X，Y=None) #模型训练，由于PCA是无监督学习，所以Y=None，没有标签。
# 2. fit_transform(self, X,Y=None)#：将模型与X进行训练，并对X进行降维处理，返回的是降维后的数据。
# 3. get_covariance(self)#获得协方差数据
# 4. get_params(self,deep=True)#返回模型的参数
# 5. get_precision(self)#计算数据精度矩阵（ 用生成模型）
# 6. inverse_transform(self, X)#将降维后的数据转换成原始数据，但可能不会完全一样
# 7. score(self, X, Y=None)#计算所有样本的log似然平均值
# 8. transform(X)#将数据X转换成降维后的数据。当模型训练好后，对于新输入的数据，都可以用transform方法来降维。
# fit_transform是fit和transform的组合，既包括了训练又包含了转换
# 必须先用fit_transform(训练数据),再transform(测试数据) , 既不能先用transform(测试数据)也不能后用fit_transform(测试数据)
# 例子
import  matplotlib.pyplot as plt
from sklearn import decomposition,datasets
iris=datasets.load_iris()#加载数据
X=iris['data']
model=decomposition.PCA(n_components=2)
#model.fit(X)
#x_nnn = model.transform(X)
X_new=model.fit_transform(X)
Maxcomponent=model.components_
ratio=model.explained_variance_ratio_
score=model.score(X)
print('降维后的数据:',X_new)
print('返回具有最大方差的成分:',Maxcomponent)
print('保留主成分的方差贡献率:',ratio)
print('所有样本的log似然平均值:',score)
print('奇异值:',model.singular_values_)
print('噪声协方差:',model.noise_variance_)
g1=plt.figure(1,figsize=(8,6))
plt.scatter(X_new[:,0],X_new[:,1],c='r',cmap=plt.cm.Set1, edgecolor='k', s=40)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('After the dimension reduction')
plt.show()
