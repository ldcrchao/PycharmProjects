#%%
# -*- coding UTF-8 -*-
'''
@Project : python学习工程文件夹
@File : Approximate_entropy.py
@Author : chenbei
@Date : 2020/12/23 13:39
'''
import numpy as np
class BaseApEn(object):
    def __init__(self, m, r):
        '''
        :param m: 子集的大小，int
        :param r: 阀值基数，0.1---0.2
        '''
        self.m = m
        self.r = r
    @staticmethod
    def _maxdist(x_i, x_j):
        return np.max([np.abs(np.array(x_i) - np.array(x_j))])
    @staticmethod
    def _biaozhuncha(U):
        if not isinstance(U, np.ndarray):
            U = np.array(U)
        return np.std(U, ddof=1)
class NewBaseApen(object):
    """新算法基类"""

    @staticmethod
    def _get_array_zeros(x):
        """
        创建N*N的0矩阵
        :param U:
        :return:
        """
        N = np.size(x, 0)
        return np.zeros((N, N), dtype=int)

    @staticmethod
    def _get_c(z, m):
        """
        计算熵值的算法
        :param z:
        :param m:
        :return:
        """
        N = len(z[0])
        # 概率矩阵C计算
        c = np.zeros((1, N - m + 1))
        if m == 2:
            for j in range(N - m + 1):
                for i in range(N - m + 1):
                    c[0, j] += z[j, i] & z[j + 1, i + 1]
        if m == 3:
            for j in range(N - m + 1):
                for i in range(N - m + 1):
                    c[0, j] += z[j, i] & z[j + 1, i + 1] & z[j + 2, i + 2]
        if m != 2 and m != 3:
            raise AttributeError('m的取值不正确！')
        data = list(filter(lambda x:x, c[0]/(N - m + 1.0)))
        if not all(data):
            return 0
        return np.sum(np.log(data)) / (N - m + 1.0)
class ApEn(BaseApEn):
    def biaozhunhua(self, U):
        self.me = np.mean(U)
        self.biao = self._biaozhuncha(U)
        return np.array([(x - self.me) / self.biao for x in U])
    def _dazhi(self, U):
        if not hasattr(self, "f"):
            self.f = self._biaozhuncha(U) * self.r
        return self.f
    def _phi(self, m, U):
        # 获取矢量列表
        x = [U[i:i + m] for i in range(len(U) - m + 1)]
        # 获取所有的比值列表
        C = [len([1 for x_j in x if self._maxdist(x_i, x_j) <= self._dazhi(U)]) / (len(U) - m + 1.0) for x_i in x]
        # 计算熵
        return np.sum(np.log(list(filter(lambda a: a, C)))) / (len(U) - m + 1.0)
    def _phi_b(self, m, U):
        # 获取矢量列表
        x = [U[i:i + m] for i in range(len(U) - m + 1)]
        # 获取所有的比值列表
        C = [len([1 for x_j in x if self._maxdist(x_i, x_j) <= self.r]) / (len(U) - m + 1.0) for x_i in x]
        # 计算熵
        return np.sum(np.log(list(filter(lambda x: x, C)))) / (len(U) - m + 1.0)
    def jinshishang(self, U):
        return np.abs(self._phi(self.m + 1, U) - self._phi(self.m, U))
    def jinshishangbiao(self, U):
        eeg = self._biaozhunhua(U)
        return np.abs(self._phi_b(self.m + 1, eeg) - self._phi_b(self.m, eeg))
class NewApEn(ApEn, NewBaseApen):
    """
    洪波等人提出的快速实用算法计算近似熵
    """

    def _get_distance_array(self, U):
        """
        获取距离矩阵
        :param U:
        :return:
        """
        z = self._get_array_zeros(U)
        fa = self._dazhi(U)
        for i in range(len(z[0])):
            z[i, :] = (np.abs(U - U[i]) <= fa) + 0
        return z

    def _get_shang(self, m, U):
        """
        计算熵值
        :param U:
        :return:
        """
        # 获取距离矩阵
        Z = self._get_distance_array(U)
        return self._get_c(Z, m)

    def hongbo_jinshishang(self, U):
        """
        计算近似熵
        :param U:
        :return:
        """
        return np.abs(self._get_shang(self.m + 1, U) - self._get_shang(self.m, U))

if __name__ == "__main__":
    U = np.array([2, 4, 6, 8, 10] * 17)
    ap = ApEn(2, 0.2)
    AEntropy = ap.jinshishang(U)  # 计算近似熵 Approximate entropy
