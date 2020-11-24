# Author：MaXiao
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np  
from sklearn.ensemble import IsolationForest
from cluster_centers import get_cluster_centers
from sklearn.preprocessing import StandardScaler, minmax_scale

class ADOA:
    """Implementation of ADOA (Anomaly Detection with Partially Observed Anomalies)"""
    def __init__(self, anomalies, unlabel, classifer, cluster_algo='kmeans', n_clusters='auto', 
                 contamination=0.01, theta=0.85, alpha='auto', beta='auto', return_proba=False, 
                 random_state=2018):
        """
        Parameters
        --------------------------
        - anomalies : 
              Observed anomaly data sets
              
        - unlabel: 
              Unlabeled data sets.
        
        - classifer: 
              A Classifer fitting weighted samples and labels to predict unlabel samples.
              
        - cluster_algo : str, {'kmeans'、'spectral'、'birch'、'dbscan'}, default = 'kmeans'
              Clustering algorithm for clustering anomaly samples.      
              
        - n_clusters: int, default=5
               The number of clusters to form as well as the number of centroids to generate.
        
        - contamination : float, range (0, 0.5).
              The proportion of outliers in the data set. 

        - theta : float, range [0, 1].
              The weights of isolation_score and similarity_score are theta and 1-theta respectively.
              
        - alpha : float, should be positive number, default = mean value of anomalies's score
              Threshold value for determining unlabel sample as potential anomaly
              
        - beta : float, should be positive number
              Threshold value for determining unlabel sample as reliable normal sample

        - return_proba : bool, default=False
              Whether return the predicted probability for positive(anomaly) class for each sample.
              Need classifer to provide predict_proba method.
        """
        dataset_scaled = StandardScaler().fit_transform(np.r_[anomalies, unlabel])
        self.anomalies = dataset_scaled[:len(anomalies), :] 
        self.unlabel = dataset_scaled[len(anomalies):, :] 
        self.contamination = contamination
        self.classifer = classifer 
        self.n_clusters = n_clusters
        self.cluster_algo = cluster_algo
        self.theta = theta 
        self.alpha = alpha 
        self.beta = beta 
        self.return_proba = return_proba 
        self.random_state = random_state
        self.centers, self.cluster_score = get_cluster_centers(self.anomalies, self.n_clusters, self.cluster_algo)
    
    def cal_weighted_score(self):
        dataset = np.r_[self.anomalies, self.unlabel]
        iforest = IsolationForest(n_estimators=100, contamination=self.contamination, 
                                  random_state=self.random_state, n_jobs=-1)
        iforest.fit(dataset)  
        # Paper：The higher is the score IS(x) (close to 1), the more likely that x being an anomaly.
        # Scikit-learn API : decision_function(X): The lower, the more abnormal.
        isolation_score = -iforest.decision_function(dataset)  
        isolation_score_scaled = minmax_scale(isolation_score)
        
        def cal_similarity_score(arr, centers=self.centers):
            min_dist = np.min([np.square(arr - center).sum() for center in centers])
            similarity_score = np.exp(-min_dist/len(arr))
            '''
            In the paper, when calculating similarity_score, min_dist is not divided by the number of features 
            (len(arr)), but when the number of features is large, the value of np.exp(min_dist) is very large, 
            so that similarity_score is close to 0, which lacks weighted meaning. Dividing by the number of 
            features helps to alleviate this phenomenon and does not affect the ordering of similarity_score.  
            '''
            return similarity_score
        similarity_score = [cal_similarity_score(arr) for arr in dataset]
        similarity_score_scaled = minmax_scale(similarity_score)
        weighted_score = self.theta * isolation_score_scaled + (1-self.theta) * similarity_score_scaled
        return weighted_score
    
    def determine_trainset(self):
        weighted_score = self.cal_weighted_score()
        min_score, max_score, median_score = [func(weighted_score) for func in (np.min, np.max, np.median)]
        anomalies_score = weighted_score[:len(self.anomalies)]
        unlabel_scores = weighted_score[len(self.anomalies):]
        
        self.alpha = np.mean(anomalies_score) if self.alpha == 'auto' else self.alpha
        self.beta = median_score if median_score < self.alpha else np.percentile(weighted_score, 45)
        assert self.beta < self.alpha, 'beta should be smaller than alpha.'
        
        # rlb:reliabel, ptt:potential
        rlb_bool, ptt_bool = unlabel_scores<=self.beta, unlabel_scores>=self.alpha
        rlb_normal, ptt_anomalies = self.unlabel[rlb_bool], self.unlabel[ptt_bool]
        rlb_normal_score, ptt_anomalies_score = unlabel_scores[rlb_bool], unlabel_scores[ptt_bool]
        rlb_normal_weight = (max_score-rlb_normal_score) / (max_score-min_score)
        ptt_anomalies_weight = ptt_anomalies_score / max_score
        
        anomalies_weight = anomalies_label = np.ones(len(self.anomalies))
        X_train = np.r_[self.anomalies, ptt_anomalies, rlb_normal] 
        weights = np.r_[anomalies_weight, ptt_anomalies_weight, rlb_normal_weight]
        y_train = np.r_[anomalies_label, np.ones(len(ptt_anomalies)), np.zeros(len(rlb_normal))].astype(int)
        return X_train, y_train, weights
    
    def predict(self):
        X_train, y_train, weights = self.determine_trainset()
        clf = self.classifer
        clf.fit(X_train, y_train, sample_weight=weights)
        y_pred = clf.predict(self.unlabel)
        if self.return_proba:
            y_prob = clf.predict_proba(self.unlabel)[:, 1]
            return y_pred, y_prob
        else:
            return y_pred
        
    def __repr__(self):
        info_1 = '1) The Observed Anomalies is divided into {:} clusters, and the calinski_harabasz_score is {:.2f}.\n'.\
        format(len(self.centers), self.cluster_score)
        
        y_train = self.determine_trainset()[1]
        rll_num = np.sum(y_train==0)
        ptt_num = sum(y_train)-len(self.anomalies)
        
        info_2 = "2) Reliable Normals's number = {:}, accounts for {:.2%} within the Unlabel dataset.\n".\
        format(ptt_num, ptt_num/len(self.unlabel))
        
        info_3 = "3) Potential Anomalies's number = {:}, accounts for {:.2%} within the Unlabel dataset.".\
        format(rll_num, rll_num/len(self.unlabel))
        
        return info_1 + info_2 + info_3
