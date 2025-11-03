from sympy import *
import numpy as np
from sklearn.model_selection import cross_validate


class Cloud_evaluater(object):

    def __init__(self):
        super(Cloud_evaluater, self).__init__()
    def Cloud_compute(self, xl):
        '''计算云滴的数字特征'''
        xl = np.array(xl)
        S2 = np.var(xl)  # 用的方差
        # S2 = np.std(xl) # 用的标准差
        Ex = np.mean(xl)
        En = np.sqrt(np.pi / 2) * np.mean(np.abs(xl - Ex))
        He = np.sqrt(np.abs(S2 * S2 - En * En))
        Ex = round(Ex, 4)
        En = round(En, 4)
        He = round(He, 4)
        return [Ex, En, He]

    def Max_value_compute(self, X):
        PN = []
        NN = []
        for i in range(len(X[0])):
            metrics_list = []
            for j in range(len(X)):
                X_i = X[j][i]
                metrics_list.append(X_i)
            PN.append(max(metrics_list))
            NN.append(min(metrics_list))
        PN = [float('{:.4f}'.format(i + 0.0005)) for i in PN]
        NN = [float('{:.4f}'.format(i - 0.0005)) for i in NN]
        return PN, NN

    def G(self, x):
        t = symbols('t')  # 定义变量
        f = (1 / np.sqrt(2 * np.pi)) * exp(-t * t / 2)
        return integrate(f, (t, -00, x))

    def Fuzzy_nearness_compute(self, N1, N2):
        x = np.abs(N2[0] - N1[0]) / (np.sqrt(N1[1] * N1[1] + N1[2] * N1[2]) + np.sqrt(N2[1] * N2[1] + N2[2] * N2[2]))
        Similarity_values = 1 / 2 + 1 / (2 * self.G(x)) - self.G(x)
        return Similarity_values

    def Assign_weight(self, Ps, Ns):
        D = []
        local_w = []
        # 计算相对相似度
        for i in range(len(Ps)):
            D_i = Ps[i] / (Ps[i] + Ns[i])
            D.append(D_i)
        # 根据相对相似度计算权重
        for i in range(len(D)):
            local_w.append(D[i] / np.sum(D))
        local_w = [float('{:.4f}'.format(i)) for i in local_w]
        return local_w



