# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
from numpy import linalg as LA


def mahal_dist(matrix):
    matrix_mean = np.mean(matrix, axis=0)
    delta = matrix - matrix_mean
    
    # 求协方差矩阵及其逆矩阵
    cov_matrix = np.cov(matrix, rowvar=False, ddof=1)
    cov_matrix_inv = LA.inv(cov_matrix)  

    # 求单个样本向量与样本中心的马氏距离
    def md_vector(vector):        
        inner_prod = np.dot(vector, cov_matrix_inv)
        inner_product = np.dot(inner_prod, vector)
        dist = np.sqrt(inner_product)
        return dist
    
    mahal_dist = np.apply_along_axis(arr=delta, axis=1, func1d=md_vector)
    return mahal_dist
