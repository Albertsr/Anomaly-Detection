import numpy as np
from numpy import linalg as LA
from sklearn.preprocessing import StandardScaler


class PCA:
    def __init__(self, matrix, k=1.0):
        self.matrix = matrix
        if isinstance(k, int):
            self.k = k
        else:
            self.k = int(np.ceil(matrix.shape[1] * k))
        
    # 将数据集标准化
    def scale(self):
        scaler = StandardScaler()
        matrix_scaled = scaler.fit_transform(self.matrix)
        return matrix_scaled
    
    # 求标准化矩阵的协方差矩阵
    def cov_matrix(self):
        # rowvar设置为False表示每列代表一个特征，每行代表一个观测值; 默认值为True
        # ddof默认值为1，表示是无偏估计
        cov_matrix = np.cov(self.scale(), rowvar=False, ddof=1)
        return cov_matrix
        
    # 求投影矩阵、各特征值的占比
    def eig(self):
        # eigenvectors的每一行即为一个特征向量
        eigenvalues, eigenvectors = LA.eig(self.cov_matrix())
        
        # 根据特征值大小对特征值、特征向量降序排列
        eigen_values = eigenvalues[np.argsort(-eigenvalues)]
        eigen_vectors = eigenvectors[np.argsort(-eigenvalues)]
        
        # 选取eigen的前K行，即为前K个特征值对应的特征向量构成的K*n型矩阵
        # 进行转置操作，使得矩阵从K*n型转置为n*K型矩阵Q，即为投影矩阵
        Q = eigen_vectors[:self.k, :].T
        return Q, eigen_values, eigen_vectors
    
    # 完成降维
    def result(self):
        Q = self.eig()[0]
        PCA_result = np.dot(self.scale(), Q)
        assert PCA_result.shape[1] == self.k, '降维后矩阵的列数应等于指定的低维度数'
        return PCA_result


class PCA_Recon_Error(PCA):
    def __init__(self, matrix, contamination=0.01):
        super(PCA_Recon_Error, self).__init__(matrix)
        self.contamination = contamination
    
    # 使用不同数量的主成分生成一系列重构矩阵
    def recon_matrix(self):
        # recon_pc_num为重构矩阵用到的top主成分数量
        def reconstruct(recon_pc_num):
            instance = PCA(self.matrix, k=recon_pc_num)
            recon_matrix = np.dot(instance.result(), (instance.eig()[0].T))
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
            square_sum = sum(np.square(vector))
            return np.sqrt(square_sum)
        # 返回单个重构矩阵生成的异常分数
        def sub_score(Rmatrix, ev_ratio):
            delta = self.scale() - Rmatrix
            score = np.apply_along_axis(vector_length, axis=1, arr=delta) * ev_ratio
            return score
        eigen_values = self.eig()[1]
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
    
    def predict(self):
        pred = [1 if i in self.anomaly_idx() else 0 for i in range(len(self.matrix))]
        return np.array(pred)
