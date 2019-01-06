# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler


class KPCA_Recon_Error:
    def __init__(self, matrix, contamination=0.01, kernel='rbf', gamma=None, random_state=2018):
        self.matrix = StandardScaler().fit_transform(matrix)
        self.contamination = contamination
        self.kernel =  kernel
        self.gamma = gamma
        self.random_state = random_state
        
    def ev_ratio(self):
        transformer = KernelPCA(n_components=None, kernel=self.kernel, gamma=self.gamma,
                                fit_inverse_transform=True, n_jobs=-1)
        transformer.fit_transform(self.matrix) 
        ev_ratio = np.cumsum(transformer.lambdas_) / np.sum(transformer.lambdas_)
        return ev_ratio
    
    def recon_matrix(self):
        def reconstruct(recon_pc_num):  
            transformer = KernelPCA(n_components=recon_pc_num, kernel=self.kernel, 
                                    gamma=self.gamma, fit_inverse_transform=True)
            X_transformed = transformer.fit_transform(self.matrix)
            # inverse_transform方法将降维后的矩阵重新映射到原来的特征空间
            recon_matrix = transformer.inverse_transform(X_transformed)
            assert recon_matrix.shape == self.matrix.shape, '重构矩阵的维度应与初始矩阵的维度一致'
            return recon_matrix
        
        # 选取不同的主成分数，生成一系列重构矩阵
        col = self.matrix.shape[1]
        recon_matrices = list(map(reconstruct, range(1, col+1)))
        
        # 检验生成的系列重构矩阵中是否存在重复
        i, j = np.random.choice(range(col), size=2, replace=False)
        assert not np.allclose(recon_matrices[i], recon_matrices[j]), '不同数量主成分生成的重构矩阵是不相同的'
        return recon_matrices
        
    def anomaly_score(self):
        # 函数vector_length用于返回向量的模
        def vector_length(vector):
            square_sum = np.sum(np.square(vector))
            return np.sqrt(square_sum)
        
        # 返回单个重构矩阵生成的异常分数
        def sub_score(recon_matrix, ev):
            delta = self.matrix - recon_matrix
            score = np.apply_along_axis(vector_length, axis=1, arr=delta) * ev
            return score
        
        # 返回所有重构矩阵生成的异常分数
        ev_ratio = self.ev_ratio()
        anomaly_scores = map(sub_score, self.recon_matrix(), ev_ratio)
        return sum(anomaly_scores)
    
    # 根据特定的污染率(contamination)返回异常分数最高的样本索引
    def anomaly_idx(self):
        idx_sort = np.argsort(-self.anomaly_score())
        anomaly_num = int(np.ceil(len(self.matrix) * self.contamination))
        anomaly_idx = idx_sort[:anomaly_num]
        return anomaly_idx
    
    # 对样本集进行预测，若判定为异常样本，则返回1，否则返回0
    def predict(self):
        pred = [1 if i in self.anomaly_idx() else 0 for i in range(len(self.matrix))]
        return np.array(pred)
