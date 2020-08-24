# Author：MaXiao
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler 


class Mahalanobis:
    """Implementation of Mahalanobis distance's variant."""

    def __init__(self, train_matrix, gamma=0.005, random_state=2018):
        """
        Parameters
        --------------------------
        - train_matrix : training set, shape = [n_samples, n_features].
        - gamma : float, default=0.005
              The proportion of abnormal samples to be eliminated in the training set.
              Increasing gamma helps to improve the sensitivity of the algorithm to abnormal samples.
        """
        self.scaler = StandardScaler().fit(train_matrix)
        self.train_matrix = self.scaler.transform(train_matrix)
        self.gamma = gamma
        self.random_state = random_state
        
    def decompose_train_matrix(self):
        pca = PCA(n_components=None, random_state=self.random_state)
        pca.fit(self.train_matrix)
        eigenvalues = pca.explained_variance_
        components = pca.components_ 
        return eigenvalues, components
    
    # the return value of compute_mahal_dist function is similar to Mahalanobis distance
    def compute_mahal_dist(self):
        eigenvalues, components = self.decompose_train_matrix()
        # get_score is used to calculate the score of each sample of the training set 
        # on a particular principal component
        def get_score(pc_idx):
            # the parameter pc_idx represents the index of the principal components 
            # components[pc_idx] represents the idx-th principal component
            inner_product = np.dot(self.train_matrix, components[pc_idx])
            score = np.square(inner_product) / eigenvalues[pc_idx]
            return score
        # calculate the score of each sample of the training set on all principal components and sum it
        mahal_dist = sum([get_score(idx) for idx in range(len(eigenvalues))])
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
    
    
class RobustPCCNew(Mahalanobis):
    """Implementation of RobustPCC Algorithm"""
    def __init__(self, train_matrix, gamma=0.005, quantile=98.99, random_state=2018):
        """
        Parameters
        --------------------------
        - train_matrix : training set, shape = [n_samples, n_features].
        - gamma : float, default=0.005
              The proportion of abnormal samples to be eliminated in the training set.
              Increasing gamma helps to improve the sensitivity of the algorithm to abnormal samples.
        - quantile: float, default=98.99
              Threshold quantile of whether it is abnormal or not.
              Increasing quantile helps to reduce the FPR(False Positive Rate) of the algorithm.
        """
        super(RobustPCCNew, self).__init__(train_matrix, gamma, random_state)
        self.quantile = quantile
    
    def decompose_remain_matrix(self):
        remain_matrix = self.eliminate_original_anomalies()
        pca = PCA(n_components=None, random_state=self.random_state)
        pca.fit(remain_matrix)
        components = pca.components_ 
        eigenvalues = pca.explained_variance_
        cumsum_ratio = np.cumsum(eigenvalues) / np.sum(eigenvalues)
        return components, eigenvalues, cumsum_ratio
    
    def compute_matrix_score(self, matrix, components, eigenvalues):
        """
        - compute_matrix_score : calculate the score of matrix on any set of eigenvalues and components.
        - get_observation_score : calculate the score of a single sample on any set of eigenvalues and components.
        - observation : a single sample.
        """
        def get_observation_score(observation):
            def sub_score(component, eigenvalue):
                inner_product = np.dot(observation, component)
                score = np.square(inner_product) / eigenvalue
                return score
            total_score = sum(map(sub_score, components, eigenvalues))
            return total_score
        matrix_scores = np.apply_along_axis(arr=matrix, axis=1, func1d=get_observation_score)
        return matrix_scores
    
    def compute_major_minor_scores(self, matrix):
        components, eigenvalues, cumsum_ratio = self.decompose_remain_matrix()   
        '''
        - compute_matrix_score：calculate the score of the given matrix corresponding to major/minor principal components.
        - major_eigen_vectors：corresponding to the first few principal components whose cumulative eigenvalues 
              account for about 50% after the eigenvalues are arranged in descending order.
        - minor_eigen_vectors：the principal components corresponding to the eigenvalue less than 0.2
        '''
        major_pc_num = len(np.argwhere(cumsum_ratio < 0.5)) + 1
        major_components = components[:major_pc_num, :]
        major_eigenvalues = eigenvalues[:major_pc_num]
        
        minor_pc_num = len(np.argwhere(eigenvalues < 0.2))
        minor_components = components[-minor_pc_num:, :]  
        minor_eigenvalues = eigenvalues[-minor_pc_num:]
        
        # calculate the score of all samples of the matrix on the major/minor principal components
        major_scores = self.compute_matrix_score(matrix, major_components, major_eigenvalues)
        minor_scores = self.compute_matrix_score(matrix, minor_components, minor_eigenvalues)
        return major_scores, minor_scores
    
    def determine_thresholds(self):
        # c1 and c2 are the anomaly thresholds corresponding to major/minor principal components, respectively.
        remain_matrix = self.eliminate_original_anomalies()
        major_scores, minor_scores = self.compute_major_minor_scores(remain_matrix)
        c1 = np.percentile(major_scores, self.quantile)
        c2 = np.percentile(minor_scores, self.quantile)
        return c1, c2
    
    # predict the testset, 1 for anomaly, 0 for normal
    def predict(self, test_matrix):
        # calculate the scores of the test set on the major/minor principal components
        # determining the deduplicated indices of abnormal samples in test set according to scores and thresholds
        c1, c2 = self.determine_thresholds()
        test_matrix_scaled = self.scaler.transform(test_matrix)
        test_major_scores, test_minor_scores = self.compute_major_minor_scores(test_matrix_scaled)  
        anomaly_indices_major = np.argwhere(test_major_scores > c1)
        anomaly_indices_minor = np.argwhere(test_minor_scores > c2) 
        test_anomaly_indices = np.union1d(anomaly_indices_major, anomaly_indices_minor)
        
        # descending arrangement of the indices of abnormal samples according to the score
        test_scores = test_major_scores + test_minor_scores 
        test_anomaly_scores = test_scores[test_anomaly_indices]
        test_anomaly_indices_desc = test_anomaly_indices[np.argsort(-test_anomaly_scores)]
    
        pred = [1 if index in test_anomaly_indices_desc else 0 for index in range(len(test_matrix))]
        return np.array(pred)
