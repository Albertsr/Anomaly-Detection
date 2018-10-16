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
        
        # variance_ratio为特征值的累计占比，作为重构误差的权重
        variance_ratio = np.cumsum(eigen_values) / np.sum(eigen_values)
        return Q, eigen_values, eigen_vectors, variance_ratio
    
    # 完成降维
    def result(self):
        PCA_result = np.dot(self.scale(), self.eig()[0])
        assert PCA_result.shape[1] == self.k, '降维后矩阵的列数应等于指定的低维度数'
        return PCA_result


class PCA_Anomaly(PCA):
    def __init__(self, matrix, contamination=0.01):
        super(PCA_Anomaly, self).__init__(matrix)
        self.contamination = contamination
        
    def anomaly_score(self):
        Q = self.eig()[0]
        ev = self.eig()[1] 
        evr = self.eig()[-1]
        major_pc_num = len(np.argwhere(evr < 0.5)) + 1
        major_pc = Q[:, :major_pc_num]
        
        temp = len(np.argwhere(ev < 0.2)) 
        minor_pc_num = temp if temp > 0 else 1
        minor_pc = Q[:, -minor_pc_num:]  
            
        def major_score(idx):
            inner_product = np.dot(self.scale(), major_pc[:, idx])
            score = np.square(inner_product) / evr[idx]
            return score
    
        def minor_score(idx):
            inner_product = np.dot(self.scale(), minor_pc[:, idx])
            score = np.square(inner_product) / evr[idx]
            return score
        
        major_scores = sum(list(map(major_score, range(major_pc_num)))) 
        minor_scores = sum(list(map(minor_score, range(minor_pc_num))))
        scores = major_scores + minor_scores 
        return scores
    
    # 根据特定的污染率返回异常样本的索引,且异常分数最高的样本索引排在前面
    def anomaly_idx(self):
        idx_sort = np.argsort(-self.anomaly_score())
        anomaly_num = int(np.ceil(len(self.matrix) * self.contamination))
        anomaly_idx = idx_sort[:anomaly_num]
        return anomaly_idx
    
    def predict(self):
        pred = [1 if i in self.anomaly_idx() else 0 for i in range(len(self.matrix))]
        return np.array(pred) 