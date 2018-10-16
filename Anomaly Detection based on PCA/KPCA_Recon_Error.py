import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler


class KPCA_Recon_Error:
    def __init__(self, matrix, contamination=0.01):
        self.matrix = matrix
        self.contamination = contamination
    
    def scale(self):
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.matrix)
        return X_scaled
    
    def transform(self):
        transformer = KernelPCA(n_components=None, kernel='rbf', gamma=2, fit_inverse_transform=True)
        transformer.fit_transform(self.scale()) 
        ev = np.cumsum(transformer.lambdas_) / np.sum(transformer.lambdas_)
        return ev
    
    def recon_matrix(self):
        def reconstruct(recon_pc_num):            
            transformer = KernelPCA(n_components=recon_pc_num, kernel='rbf', gamma=2, fit_inverse_transform=True)
            X_transformed = transformer.fit_transform(self.scale())
            # inverse_transform方法将降维后的矩阵重新映射到原来的特征空间
            recon_matrix = transformer.inverse_transform(X_transformed)
            #assert recon_matrix.shape == self.matrix.shape
            return recon_matrix
        
        col = self.matrix.shape[1]
        Recon_Matrices = list(map(reconstruct, range(1, col+1)))
        assert len(Recon_Matrices) == col
        
        # 检验生成的系列重构矩阵中是否存在重复
        i, j = np.random.choice(range(col), size=2, replace=False)
        assert not np.allclose(Recon_Matrices[i], Recon_Matrices[j])
        return Recon_Matrices
        
    
    def anomaly_score(self):
        # 返回单个重构矩阵生成的异常分数
        def sub_score(Rmatrix, ev_ratio):
            def vector_length(vector):
                square_sum = sum(np.square(vector))
                return np.sqrt(square_sum)
    
            delta = self.scale() - Rmatrix
            score = np.apply_along_axis(vector_length, axis=1, arr=delta) * ev_ratio
            return score
        
        # 返回所有重构矩阵生成的异常分数
        ev = self.transform()
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