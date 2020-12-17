# Author：MaXiao
# E-mail：maxiaoscut@aliyun.com

import numpy as np
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler 

def cal_mahal_dist_variant(matrix):
    matrix = StandardScaler().fit_transform(matrix)
    cov_matrix = np.cov(matrix, rowvar=False, ddof=1)
    eigen_values, eigen_vectors = LA.eig(cov_matrix)
    
    def get_score(idx):
        inner_product = np.dot(matrix, eigen_vectors[:, idx])
        score = np.square(inner_product) / eigen_values[idx]
        return score
    mahal_dist_variant = sum([get_score(i) for i in range(len(eigen_values))])
    assert len(mahal_dist_variant) == len(matrix)
    return mahal_dist_variant
