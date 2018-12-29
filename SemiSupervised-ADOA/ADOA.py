# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np  
from sklearn.ensemble import IsolationForest
from cluster_centers import get_cluster_centers


class ADOA:
    def __init__(self, anomalies, unlabel, clusters='auto', cluster_algo='kmeans', cluster_param_grid='auto', cluster_verbose=True,
                 itrees=100, contamination=0.01, theta=0.8, alpha='auto', beta='auto', random_state=2018):
        self.anomalies = anomalies
        self.unlabel = unlabel
        self.clusters = clusters # 聚类簇数可以预先指定，也可以由cluster_centers自动确定最佳聚类簇数
        self.cluster_algo = cluster_algo # cluster_algo可以选取'spectral'、'birch'、'dbscan'、'kmeans'，默认取'kmeans'
        self.cluster_param_grid = cluster_param_grid
        self.cluster_verbose = cluster_verbose
        self.itrees = itrees
        self.contamination = contamination
        self.theta = theta
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state
        self.centers = get_cluster_centers(self.anomalies, self.clusters, self.cluster_algo, self.cluster_param_grid)
    
    def cluster_info(self):
        if self.cluster_verbose:
            print('clusters_number: {:}'.format(len(self.centers)))
        
    def similarity_score(self):
        # 与异常值的各簇中心的最近距离dist决定了最终的similarity score，且两者成反比
        def compute_ss(x, centers=self.centers):
            dist = np.min([np.sum(np.square(x - center)) for center in centers])     
            similarity_score = np.exp(-dist)
            return similarity_score
        if self.unlabel.shape[0] == 1:
            return compute_ss(self.unlabel)
        else:
            return np.array(list(map(compute_ss, self.unlabel)))   
        
    def isolation_score(self):
        iforest = IsolationForest(n_estimators=self.itrees, contamination=self.contamination, 
                                  random_state=self.random_state, n_jobs=-1, behaviour="new")
        iforest.fit(self.unlabel)  
        # 前面的负号不能丢;
        # 论文原文：The higher is the score IS(x) (close to 1), the more likely that x being an anomaly
        isolation_score = - iforest.decision_function(self.unlabel)
        return isolation_score
    
    def total_score(self):
        # total_score越大，样本点越有可能是异常点
        isolation_score = np.dot(self.theta, self.isolation_score())
        similarity_score = np.dot(1-self.theta, self.similarity_score())
        total_score = isolation_score + similarity_score
        return total_score
    
    def classify(self):
        if self.alpha == 'auto' and self.beta == 'auto':
            self.alpha = np.percentile(self.total_score(), 55)
            self.beta = np.percentile(self.total_score(), 45)
        def clf(x):
            if x >= self.alpha:
                # 'potential anomalies'
                return 1
            elif x <= self.beta:
                # 'reliable normal samples' 
                return 0 
            else:
                return 0.5
        # U中所有样本的分类结果
        clf_result = np.array(list(map(clf, self.total_score())))
        # 确定reliable_normal与potential_anomalies
        reliable_normal = self.unlabel[clf_result==0]
        potential_anomalies = self.unlabel[clf_result==1]
        return clf_result, reliable_normal, potential_anomalies
        
    def set_weight(self):
        # 将原有异常样本的权重全部设置为1
        observed_anomalies_weight = np.ones(self.anomalies.shape[0])
        # 求出无标签样本集的最小与最大分数
        TS_MIN, TS_MAX = np.min(self.total_score()), np.max(self.total_score())
        # 设置可靠正样本的权重
        reliable_normal_score = self.total_score()[self.classify()[0]==0]
        reliable_normal_weight = (TS_MAX - reliable_normal_score) / (TS_MAX - TS_MIN)
        
        # 设置潜在异常样本的权重
        potential_anomalies_score = self.total_score()[self.classify()[0]==1]
        potential_anomalies_weight = potential_anomalies_score / TS_MAX
        return reliable_normal_weight, potential_anomalies_weight
