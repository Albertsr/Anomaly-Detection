# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCA_Recon_Error:
    def __init__(self, matrix, contamination=0.01, random_state=2018):
        self.matrix = StandardScaler().fit_transform(matrix)
        self.contamination = contamination
        self.random_state = random_state
            
    def ev_ratio(self):
        pca = PCA(n_components=None, random_state=self.random_state)
        pca_result = pca.fit_transform(self.matrix)
        # explained_variance_属性返回降序排列的协方差矩阵的特征值
        eigenvalues = pca.explained_variance_
        # ev_ratio为特征值的累计占比，作为不同数量主成分对应的重构误差的权重
        ev_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        return ev_ratio
        
    # 使用不同数量的主成分生成一系列重构矩阵
    def recon_matrix(self):
        # 参数recon_pc_num为重构矩阵用到的top主成分数量
        def reconstruct(recon_pc_num):
            pca_recon = PCA(n_components=recon_pc_num, random_state=self.random_state)
            pca_reduction = pca_recon.fit_transform(self.matrix)
            # inverse_transform方法能返回重构矩阵
            recon_matrix = pca_recon.inverse_transform(pca_reduction)
            assert recon_matrix.shape == self.matrix.shape, '重构矩阵的维度应与初始矩阵的维度一致'
            return recon_matrix
        
        # 生成一系列重构矩阵
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
        
        # 返回单个重构矩阵对所有样本生成的异常分数
        def sub_score(recon_matrix, ev):
            delta = self.matrix - recon_matrix
            score = np.apply_along_axis(vector_length, axis=1, arr=delta) * ev
            return score
        
        # 返回所有重构矩阵生成的异常分数并汇总
        ev_ratio = self.ev_ratio()
        anomaly_scores = map(sub_score, self.recon_matrix(), ev_ratio)
        return sum(anomaly_scores)
    
    # 根据特定的污染率(contamination)返回异常分数最高的样本索引
    def anomaly_idx(self):
        # 根据异常分数降序排列，并返回对应的索引
        idx_sort = np.argsort(-self.anomaly_score())
        # 根据contamination确定需要返回的异常样本数量
        anomaly_num = int(np.ceil(len(self.matrix) * self.contamination))
        anomaly_idx = idx_sort[:anomaly_num]
        return anomaly_idx
    
    # 对样本集进行预测，若判定为异常样本，则返回1，否则返回0
    def predict(self):
        pred = [1 if i in self.anomaly_idx() else 0 for i in range(len(self.matrix))]
        return np.array(pred)
