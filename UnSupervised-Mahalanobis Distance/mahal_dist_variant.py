# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler 


def mahal_dist_variant(matrix):
    # 将数据集标准化
    matrix = StandardScaler().fit_transform(matrix)
    # 对数据集进行主成分分析
    cov_matrix = np.cov(matrix, rowvar=False, ddof=1)
    eigen_values, eigen_vectors = LA.eig(cov_matrix)
        
    # 函数get_score用于返回数据集在单个主成分上的分数
    # 参数pc_idx表示主成分的索引
    def get_score(pc_idx):
        # eigen_vectors.T[pc_idx]表示第idx个主成分构成的列向量
        inner_product = np.dot(matrix, eigen_vectors[pc_idx])
        score = np.square(inner_product) / eigen_values[pc_idx]
        return score
    # 返回训练集每一个样本在所有主成分上的分数，并分别求和
    mahal_dist = sum(map(get_score, range(len(eigen_values))))
    return mahal_dist
