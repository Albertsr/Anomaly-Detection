# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr


import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCA_Recon_Error:
    def __init__(self, matrix, contamination=0.01, random_state=2018):
        self.matrix = matrix
        self.contamination = contamination
        self.random_state = random_state
    
    def scale(self):
        scaler = StandardScaler()
        matrix_scaled = scaler.fit_transform(self.matrix)
        return matrix_scaled
        
    def ev_ratio(self):
        pca_ = PCA(n_components=None, random_state=self.random_state)
        pca_result = pca_.fit_transform(self.scale())
        eigenvalues = pca_.explained_variance_
        
        # ev_ratio为特征值的累计占比，作为重构误差的权重
        ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        return ratio
        
    # 使用不同数量的主成分生成一系列重构矩阵
    def recon_matrix(self):
        # 参数recon_pc_num为重构矩阵用到的top主成分数量
        def reconstruct(recon_pc_num):
            pca_recon = PCA(n_components=recon_pc_num, random_state=self.random_state)
            pca_reduction = pca_recon.fit_transform(self.scale())
            # inverse_transform方法能返回重构矩阵
            recon_matrix = pca_recon.inverse_transform(pca_reduction)
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
        
        # 返回所有重构矩阵生成的异常分数
        ev_ratio = self.ev_ratio()
        scores = list(map(sub_score, self.recon_matrix(), ev_ratio)) 
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
