# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
from numpy import linalg as LA


class PCA_SVD:
    # 参数n_components为保留的主成分数
    def __init__(self, matrix, n_components=None):
        self.matrix = matrix
        self.n_components = matrix.shape[1] if n_components==None else n_components
    
    # 自定义标准化方法
    def scale(self):
        def scale_vector(vector):
            delta = vector - np.mean(vector)
            std = np.std(vector, ddof=0)
            return delta / std
        matrix_scaled = np.apply_along_axis(arr=self.matrix, func1d=scale_vector, axis=0)
        return matrix_scaled
     
    # 对标准化后的矩阵进行奇异值分解    
    def matrix_svd(self):
        # 令A为m*n型矩阵，则U、V分别为m阶、n阶正交矩阵
        # U的每一个列向量都是A*A.T的特征向量，也称为左奇异向量
        # V的每一个行向量都是A.T*A的特征向量，也称为右奇异向量
        # sigma是由k个降序排列的奇异值构成的向量，其中k = min(matrix.shape)
        U, sigma, V =  LA.svd(self.scale()) 
        
        # 非零奇异值的个数不会超过原矩阵的秩，从而不会超过矩阵维度的最小值
        assert len(sigma) == min(self.matrix.shape)
        return U, sigma, V 
    
    # 通过矩阵V进行PCA，返回最终降维后的矩阵
    def pca_result(self):
        sigma, V = self.matrix_svd()[1], self.matrix_svd()[2]
        # Q为投影矩阵，由V的前n_components个行向量转置后得到
        Q = V[:self.n_components, :].T
        # 计算标准化后的矩阵在Q上的投影，得到PCA的结果
        matrix_pca = np.dot(self.scale(), Q)
        # matrix_pca的列数应等于保留的主成分数
        assert matrix_pca.shape[1] == self.n_components
        return matrix_pca
    
    
class PCA_Recon_Error(PCA_SVD):
    def __init__(self, matrix, n_components=None, contamination=0.01):
        super(PCA_Recon_Error, self).__init__(matrix, n_components)
        self.contamination = contamination
    
    # 使用不同数量的主成分生成一系列重构矩阵
    def recon_matrix(self):
        # recon_pc_num为重构矩阵用到的top主成分数量
        def reconstruct(recon_pc_num):
            instance = PCA_SVD(self.matrix, n_components=recon_pc_num)
            V = instance.matrix_svd()[-1]
            
            # instance.pca_result()为PCA降维的结果
            # V[:recon_pc_num, :]：以recon_pc_num个主成分为行向量构成的矩阵
            # recon_matrix为重构矩阵
            recon_matrix = np.dot(instance.pca_result(), V[:recon_pc_num, :])
            assert recon_matrix.shape == self.matrix.shape, '重构矩阵的维度应与初始矩阵的维度一致'
            return recon_matrix
        
        col = self.matrix.shape[1]
        recon_matrices = list(map(reconstruct, range(1, col+1)))
        
        # 检验生成的系列重构矩阵中是否存在重复
        i, j = np.random.choice(range(col), size=2, replace=False)
        assert not np.allclose(recon_matrices[i], recon_matrices[j]), '不同数量主成分生成的重构矩阵是不相同的'
        return recon_matrices
        
    # 返回最终的异常分数  
    def anomaly_score(self):
        # 函数vector_length用于返回向量的模
        def vector_length(vector):
            square_sum = np.sum(np.square(vector))
            return np.sqrt(square_sum)
        
        # 返回单个重构矩阵生成的异常分数
        def sub_score(recon_matrix, ev_ratio):
            delta = self.scale() - recon_matrix
            score = np.apply_along_axis(vector_length, axis=1, arr=delta) * ev_ratio
            return score
        
        # numpy.svd方法返回的奇异值已降序排列
        single_value = self.matrix_svd()[1]
        # 奇异值的平方为协方差矩阵的特征值
        eigen_values = np.square(single_value)
        # ev为特征值的累计占比，作为重构误差的权重
        ev = np.cumsum(eigen_values) / np.sum(eigen_values)
        # 返回所有重构矩阵生成的异常分数
        scores = list(map(sub_score, self.recon_matrix(), ev)) 
        return sum(scores)
    
    # 根据特定的污染率返回异常样本的索引,且异常分数最高的样本索引排在前面
    def anomaly_idx(self):
        idx_sort = np.argsort(-self.anomaly_score())
        anomaly_num = int(np.ceil(len(self.matrix) * self.contamination))
        anomaly_idx = idx_sort[:anomaly_num]
        return anomaly_idx
    
    # 对样本集进行预测，若判定为异常样本，则返回1，否则返回0
    def predict(self):
        pred = [1 if i in self.anomaly_idx() else 0 for i in range(len(self.matrix))]
        return np.array(pred)
