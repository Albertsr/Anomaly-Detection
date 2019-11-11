# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 


class Mahalanobis:
    # gamma is the proportion of abnormal samples to be eliminated in the training set
    # the default value of gamma in the paper is 0.005
    def __init__(self, train_matrix, gamma=0.005, random_state=2018):
        self.train_matrix = StandardScaler().fit_transform(train_matrix)
        self.gamma = gamma
        self.random_state = random_state
        
    def decompose_train_matrix(self):
        pca = PCA(n_components=None, random_state=self.random_state)
        pca.fit(self.train_matrix)
        eigen_values = pca.explained_variance_
        eigen_vectors = pca.components_ 
        return eigen_values, eigen_vectors
    
    # the return value of compute_mahal_dist function is similar to Mahalanobis distance
    def compute_mahal_dist(self):
        eigen_values, eigen_vectors = self.decompose_train_matrix()
        # get_score is used to calculate the score of each sample of the training set on a particular principal component
        def get_score(pc_idx):
            # the parameter pc_idx represents the index of the principal components 
            # eigen_vectors.T[pc_idx] represents the idx-th principal component
            inner_product = np.dot(self.train_matrix, eigen_vectors.T[pc_idx])
            score = np.square(inner_product) / eigen_values[pc_idx]
            return score
        # calculate the score of each sample of the training set on all principal components and sum it
        mahal_dist = sum([get_score(i) for i in range(len(eigen_values))])
        return mahal_dist
    
    # return the indices of the anomaly samples in the training set
    def search_original_anomaly_indices(self):
        indices_sort = np.argsort(-self.compute_mahal_dist())
        anomaly_num = int(np.ceil(len(self.train_matrix) * self.gamma))
        original_anomaly_indices = indices_sort[:anomaly_num]
        return original_anomaly_indices
    
    def eliminate_original_anomalies(self):  
        original_anomaly_indices = self.search_original_anomaly_indices()
        train_matrix_indices = range(len(self.train_matrix))
        condition = np.isin(train_matrix_indices, original_anomaly_indices, invert=True)
        remain_matrix = self.train_matrix[condition] # np.extract(condition, self.train_matrix)
        return remain_matrix
    
    
class RobustPCC(Mahalanobis):
    # increasing gamma helps to improve the sensitivity of the algorithm to abnormal samples
    # increasing quantile helps to reduce the FPR of the algorithm
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
    
    # compute_matrix_score function is used to calculate the score of matrix on any set of eigenvalues and eigenvectors
    def compute_matrix_score(self, matrix, eigen_vectors, eigen_values):
        # get_observation_ score is used to calculate the score of a single sample on any set of eigenvalues and eigenvectors
        def get_observation_score(observation):
            def sub_score(eigen_vector, eigen_value):
                inner_product = np.dot(observation, eigen_vector)
                score = np.square(inner_product) / eigen_value
                return score
            total_score = sum(map(sub_score, eigen_vectors, eigen_values))
            return total_score
        matrix_scores = np.apply_along_axis(arr=matrix, axis=1, func1d=get_observation_score)
        return matrix_scores
    
    # compute_matrix_score function calculate the score of the given matrix corresponding to major/minor principal components
    def compute_major_minor_scores(self, matrix):
        eigen_values, eigen_vectors, cumsum_ratio = self.decompose_remain_matrix()   
              
        # major_eigen_vectors refers to the eigenvectors corresponding to the first few eigenvalues 
        # whose cumulative eigenvalues account for about 50% after the eigenvalues are arranged in descending order
        major_pc_num = len(np.argwhere(cumsum_ratio < 0.5)) + 1
        major_eigen_vectors = eigen_vectors[:major_pc_num, :]
        major_eigen_values = eigen_values[:major_pc_num]
        
        # minor_eigen_vectors is the eigenvectors corresponding to the eigenvalue less than 0.2
        minor_pc_num = len(np.argwhere(eigen_values < 0.2))
        minor_eigen_vectors = eigen_vectors[-minor_pc_num:, :]  
        minor_eigen_values = eigen_values[-minor_pc_num:]
        
        # calculate the score of all samples of the matrix on the major/minor principal components
        matrix_major_scores = self.compute_matrix_score(matrix, major_eigen_vectors, major_eigen_values)
        matrix_minor_scores = self.compute_matrix_score(matrix, minor_eigen_vectors, minor_eigen_values)
        return matrix_major_scores, matrix_minor_scores
         
    def search_test_anomaly_indices(self):
        # c1 and c2 are the anomaly thresholds corresponding to major/minor principal components, respectively.
        remain_matrix = self.eliminate_original_anomalies()
        matrix_major_scores, matrix_minor_scores = self.compute_major_minor_scores(remain_matrix)
        c1 = np.percentile(matrix_major_scores, self.quantile)
        c2 = np.percentile(matrix_minor_scores, self.quantile)
        
        # calculate the scores of the test set on the major/minor principal components
        # determining the deduplicated indices of abnormal samples in test set according to scores and thresholds
        test_major_score, test_minor_score = self.compute_major_minor_scores(self.test_matrix)  
        anomaly_indices_major = np.argwhere(test_major_score > c1)
        anomaly_indices_minor = np.argwhere(test_minor_score > c2) 
        test_anomaly_indices = np.union1d(anomaly_indices_major, anomaly_indices_minor)
        
        # descending arrangement of the indices of abnormal samples according to the score
        test_scores = test_major_score + test_minor_score 
        test_anomaly_scores = test_scores[test_anomaly_indices]
        test_anomaly_indices_desc = test_anomaly_indices[np.argsort(-test_anomaly_scores)]
        return test_anomaly_indices_desc
    
    # predict the test set, 1 for anomaly, 0 for normal
    def predict(self):
        test_anomaly_indices = self.search_test_anomaly_indices()        
        pred = [1 if i in test_anomaly_indices else 0 for i in range(len(self.test_matrix))]
        return np.array(pred)
