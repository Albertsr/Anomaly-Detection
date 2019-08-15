# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 


class Mahalanobis:
    # gamma为训练集中待剔除的异常样本比例，论文默认取0.005
    def __init__(self, train_matrix, gamma=0.005, random_state=2018):
        self.train_matrix = StandardScaler().fit_transform(train_matrix)
        self.gamma = gamma
        self.random_state = random_state
        
    def decompose_train_matrix(self):
        pca = PCA(n_components=None, random_state=self.random_state)
        pca.fit(self.train_matrix)
        # explained_variance_、components_ 分别返回降序排列的特征值与特征向量
        eigen_values = pca.explained_variance_
        eigen_vectors = pca.components_ 
        return eigen_values, eigen_vectors
    
    # 论文里明确指出compute_mahal_dist函数的返回值等价于为马氏距离
    # 经过测试，compute_mahal_dist函数对样本异常程度的预估与马氏距离的大小关系完全一致
    def compute_mahal_dist(self):
        eigen_values, eigen_vectors = self.decompose_train_matrix()
        # 函数get_score用于返回训练集每一个样本在特定主成分上的分数
        # 参数pc_idx表示主成分的索引
        def get_score(pc_idx):
            # eigen_vectors.T[pc_idx]表示第idx个主成分构成的列向量
            inner_product = np.dot(self.train_matrix, eigen_vectors.T[pc_idx])
            score = np.square(inner_product) / eigen_values[pc_idx]
            return score
        # 返回训练集每一个样本在所有主成分上的分数，并分别求和
        mahal_dist = sum([get_score(i) for i in range(len(eigen_values))])
        return mahal_dist
    
    # 返回异常样本的索引，此函数本身也可用于异常检测
    def search_original_anomaly_indices(self):
        indices_sort = np.argsort(-self.compute_mahal_dist())
        anomaly_num = int(np.ceil(len(self.train_matrix) * self.gamma))
        anomaly_indices = indices_sort[:anomaly_num]
        return anomaly_indices
    
    # 删除较异常的样本
    def eliminate_original_anomalies(self):  
        anomaly_indices = self.search_original_anomaly_indices()
        matrix_indices = range(len(self.train_matrix))
        condition = np.isin(matrix_indices, anomaly_indices, invert=True)
        remain_matrix = self.train_matrix[condition]
        return remain_matrix
    
    
class RobustPCC(Mahalanobis):
    # quantile为确定阈值的分位点，论文默认取98.99
    # 在样本数较多的情况下，可适当提高gamma与quantile的取值，以保证PCC的鲁棒性，降低FPR
    def __init__(self, train_matrix, test_matrix, gamma=0.005, quantile=98.99, random_state=2018):
        super(RobustPCC, self).__init__(train_matrix, gamma, random_state)
        self.test_matrix = StandardScaler().fit_transform(test_matrix)
        self.quantile = quantile
    
    def decompose_remain_matrix(self):
        remain_matrix = self.eliminate_original_anomalies()
        pca = PCA(n_components=None, random_state=self.random_state)
        pca.fit(remain_matrix)
        eigen_vectors = pca.components_ 
        eigen_values = pca.explained_variance_
        cumsum_ratio = np.cumsum(eigen_values) / np.sum(eigen_values)
        return eigen_values, eigen_vectors, cumsum_ratio
    
    # compute_matrix_score函数用于返回matrix在一组特征值、特征向量上的分数
    # matrix的每一行表示一个样本
    def compute_matrix_score(self, matrix, eigen_vectors, eigen_values):
        # 构建get_observation_score子函数，用于返回单个样本在任意一组特征值、特征向量上的分数
        def get_observation_score(observation):
            # 子函数sub_score用于返回单个样本在特定单个特征值向量上的总分数
            def sub_score(eigen_vector, eigen_value):
                inner_product = np.dot(observation, eigen_vector)
                score = np.square(inner_product) / eigen_value
                return score
            # 返回单个样本在所有特征值向量上对应的总分数
            total_score = sum(map(sub_score, eigen_vectors, eigen_values))
            return total_score
        matrix_scores = np.apply_along_axis(arr=matrix, axis=1, func1d=get_observation_score)
        return matrix_scores
    
    # 构建compute_major_minor_scores函数，返回给定matrix中所有样本在major/minor principal components对应的分数
    def compute_major_minor_scores(self, matrix):
        eigen_values, eigen_vectors, cumsum_ratio = self.decompose_remain_matrix()   
               
        # major principal components是指特征值降序排列后，累计特征值之和约占50%的前几个特征值对应的特征向量
        # major_pc_num为major principal components的个数
        # major_eigen_vectors，major_eigen_values分别为major principal components对应的特征向量与特征值
        major_pc_num = len(np.argwhere(cumsum_ratio < 0.5)) + 1
        major_eigen_vectors = eigen_vectors[:major_pc_num, :]
        major_eigen_values = eigen_values[:major_pc_num]
        
        # minor principal components是特征值小于0.2对应的特征向量
        minor_pc_num = len(np.argwhere(eigen_values < 0.2))
        minor_eigen_vectors = eigen_vectors[-minor_pc_num:, :]  
        minor_eigen_values = eigen_values[-minor_pc_num:]
        
        # 返回矩阵所有样本在major/minor principal components对应的分数
        matrix_major_scores = self.compute_matrix_score(matrix, major_eigen_vectors, major_eigen_values)
        matrix_minor_scores = self.compute_matrix_score(matrix, minor_eigen_vectors, minor_eigen_values)
        return matrix_major_scores, matrix_minor_scores
         
    def search_test_anomaly_indices(self):
        # c1、c2分别为major/minor principal components对应的阈值
        remain_matrix = self.eliminate_original_anomalies()
        matrix_major_scores, matrix_minor_scores = self.compute_major_minor_scores(remain_matrix)
        c1 = np.percentile(matrix_major_scores, self.quantile)
        c2 = np.percentile(matrix_minor_scores, self.quantile)
        
        # 求test_matrix在major/minor principal components上对应的分数
        test_major_score, test_minor_score = self.compute_major_minor_scores(self.test_matrix)  
        # 根据阈值判定test_matrix中的异常样本
        anomaly_indices_major = np.argwhere(test_major_score > c1)
        anomaly_indices_minor = np.argwhere(test_minor_score > c2)  
        # 返回去重的异常样本索引
        anomaly_indices = np.union1d(anomaly_indices_major, anomaly_indices_minor)
        
        # 根据异常总分对异常样本索引进行降序排列
        total_scores =  test_major_score + test_minor_score 
        anomaly_scores = total_scores[anomaly_indices]
        anomaly_indices_desc = anomaly_indices[np.argsort(-anomaly_scores)]
        return anomaly_indices_desc
    
    def predict(self):
        anomaly_indices = self.test_anomaly_indices()        
        pred = [1 if i in anomaly_indices else 0 for i in range(len(self.test_matrix))]
        assert sum(pred) == len(anomaly_indices)
        return np.array(pred)
