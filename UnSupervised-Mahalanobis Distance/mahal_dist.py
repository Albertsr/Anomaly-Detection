# Author：MaXiao
# E-mail：maxiaoscut@aliyun.com

import numpy as np
from numpy import linalg as LA

def cal_mahal_dist(matrix):
    matrix_center = np.mean(matrix, axis=0)
    delta = matrix - matrix_center
    
    # calculate the covariance matrix and its inverse matrix
    cov_matrix = np.cov(matrix, rowvar=False, ddof=1)
    cov_matrix_inv = LA.inv(cov_matrix)  
    
    # calculate the Mahalanobis distance between a single vector and the center of the dataset
    def md_vector(vector):        
        inner_prod = np.dot(vector, cov_matrix_inv)
        dist = np.sqrt(np.dot(inner_prod, vector))
        return dist

    mahal_dist = np.apply_along_axis(arr=delta, axis=1, func1d=md_vector)
    assert len(mahal_dist) == len(matrix)
    return mahal_dist
