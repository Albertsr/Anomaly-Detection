# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np  
from sklearn.ensemble import IsolationForest
from cluster_centers import get_cluster_centers
from sklearn.preprocessing import StandardScaler, minmax_scale

class ADOA:
    def __init__(self, anomalies, unlabel, classifer, return_proba=True, n_clusters='auto', cluster_algo='kmeans', 
                 contamination=0.02, theta=0.85, alpha='auto', beta='auto', random_state=2018):
        scaler = StandardScaler()
        scaled_dataset = scaler.fit_transform(anomalies.append(unlabel))
        self.anomalies = scaled_dataset.iloc[:len(anomalies), :] 
        self.unlabel = scaled_dataset.iloc[len(anomalies):, :] 
        self.n_clusters = n_clusters # 聚类簇数可以预先指定，也可以由get_cluster_centers自动确定最佳聚类簇数
        self.classifer = classifer # 选取的分类器
        self.return_proba = return_proba # 布尔型参数，是否返回样本取正的后验概率
        self.cluster_algo = cluster_algo # 聚类算法，可以选取'spectral'、'birch'、'dbscan'、'kmeans'，默认取'kmeans'
        self.contamination = contamination # contamination为预估的U中异常样本(即正样本)比例
        self.theta = theta # isolation_score、similarity_score的加权系数分别为theta、1-theta
        self.alpha = alpha # 论文默认取已知异常样本(P集)的total_score均值，作为无标签样本是否为potential anomalies的阈值
        self.beta = beta # 判定无标签样本是否为reliable normal的阈值
        self.random_state = random_state
        self.centers = get_cluster_centers(self.anomalies, self.n_clusters, self.cluster_algo) # 返回聚类的簇中心
    
    # 返回算法在给定数据集下生成的基本信息
    def __repr__(self):
        info_1 = '1) The positives(observed_anomalies) is divided into {:} clusters.\n'
        info_2 = '2) The thresholds : alpha = {:.6f}, beta = {:.6f}\n'
        info_3 = "3) reliable normal's number = {:}, potential anomalies's number = {:}\n"
        info_4 = "4) The final negative's number = {:}, positive's number = {:}"
        info = (info_1 + info_2 + info_3 + info_4)
        a, _, b, _ = self.classify_unlabel()
        y_train = self.weighted_trainset()[1]
        return info.format(len(self.centers), self.alpha, self.beta, len(a), len(b),\
                           np.sum(y_train==0), np.sum(y_train==1))
    
    # 返回数据集中所有样本的total_score
    def get_total_score(self):
        dataset = np.r_[self.anomalies, self.unlabel]
        
        # 计算isolation_score
        iforest = IsolationForest(n_estimators=120, contamination=self.contamination, 
                                  random_state=self.random_state, n_jobs=-1, behaviour="new")
        iforest.fit(dataset)  
        # ADOA论文原文：The higher is the score IS(x) (close to 1), the more likely that x being an anomaly
        # Sklearn document：score_samples(X): Opposite of the anomaly score defined in the original paper.
        isolation_score = -iforest.score_samples(dataset)  # 负号不能丢
 
        # 计算similarity_score
        def get_similarity_score(x, centers=self.centers):
            # 计算样本与P集各簇中心的最短距离，再计算最终的similarity score，且两者成反比
            min_dist = np.min([np.square(x - center).sum() for center in centers])   
            '''
            计算similarity_score时，论文原文中min_dist没有除以特征数，但实际数据表明：当特征数较多时，
            即使数据已经标准化，自然底数e的指数幂构成的分母也可能极大，similarity_score的最大值也接近于0，
            缺乏加权意义，而除以特征数self.anomalies.shape[1]有助于缓解此现象，且不影响距离的相对大小关系
            '''
            similarity_score = np.exp(-min_dist/self.anomalies.shape[1])
            return similarity_score
        similarity_score = np.array(list(map(get_similarity_score, dataset)))
        # 对similarity_score进一步MIN_MAX标准化，与isolation_score保持一致的[0, 1]区间
        similarity_score = minmax_scale(similarity_score)     
        
        # 计算total_score，对isolation_score应赋予较大的权重，self.theta默认取0.85
        total_score = isolation_score * self.theta + similarity_score * (1-self.theta)
        return total_score
    
    # 对无标签样本集U进行分类，从中筛选出potential anomalies、reliable normal
    def classify_unlabel(self):
        total_score = self.get_total_score()
        # observed_anomalies_score、unlabel_scores分别为U集、P集对应的total_score
        observed_anomalies_score = total_score[:len(self.anomalies)]
        unlabel_scores = total_score[len(self.anomalies):]
        
        if self.alpha == 'auto' and self.beta == 'auto':
            # 论文将alpha的取值默认设定为已知异常样本total_score的均值
            self.alpha = np.mean(observed_anomalies_score)
            
            # beta为判定无标签样本是否为reliable normal samples的阈值
            # 如果所有样本total_score的中位数np.median(total_score)小于alpha，则视其为beta，否则取45分位点处的值
            if np.median(total_score) < self.alpha:
                self.beta = np.median(total_score)
            else:
                self.beta = np.percentile(total_score, 45)
            # beta的取值应小于alpha，否则输出warning
            if self.beta > self.alpha:
                warning = 'beta({:.6f}) is bigger than alpha({:.6f}), you should properly decrease the value of beta.'
                print(warning.format(self.beta, self.alpha))

        def clf(score):
            if score >= self.alpha:
                # 'potential anomalies'
                return 1
            elif score <= self.beta:
                # 'reliable normal samples' 
                return 0 
            else:
                return 0.5
            
        # clf_result为U中所有样本的分类结果
        clf_result = np.array(list(map(clf, unlabel_scores)))
        # 确定reliable_normal及其分数
        reliable_normal = self.unlabel[clf_result==0]
        reliable_normal_score = unlabel_scores[clf_result==0]
        # 确定potential_anomalies及其分数
        potential_anomalies = self.unlabel[clf_result==1]
        potential_anomalies_score = unlabel_scores[clf_result==1]
        return reliable_normal, reliable_normal_score, potential_anomalies, potential_anomalies_score
        
    def weighted_trainset(self): 
        total_score = self.get_total_score()
        reliable_normal, reliable_normal_score, potential_anomalies, potential_anomalies_score = self.classify_unlabel()
        
        # 求出total_score的最小与最大值
        score_min, score_max = np.min(total_score), np.max(total_score)
        # 设置reliable_normal的权重
        reliable_normal_weight = (score_max-reliable_normal_score) / (score_max-score_min)
        # 设置潜在异常样本(potential_anomalies)的权重
        potential_anomalies_weight = potential_anomalies_score / score_max
        # 将原有异常样本的权重全部设置为1
        observed_anomalies_weight = np.ones(len(self.anomalies))
        
        # 生成训练数据：将已知的anomalies, potential_anomalies, reliable_normal在axis=0方向上予以整合
        X_train = np.r_[self.anomalies, potential_anomalies, reliable_normal] 
        # 生成权重
        weights = np.r_[observed_anomalies_weight, potential_anomalies_weight, reliable_normal_weight]
        
        # 生成标签：anomalies, potential_anomalies的标签均为1，reliable_normal的标签为0
        observed_anomalies_label = observed_anomalies_weight
        potential_anomalies_label = np.ones(len(potential_anomalies))
        reliable_normal_label = np.zeros(len(reliable_normal))
        y_train = np.r_[observed_anomalies_label, potential_anomalies_label, reliable_normal_label]
        return X_train, y_train, weights
    
    def predict(self):
        X_train, y_train, weights = self.weighted_trainset()
        clf = self.classifer
        clf.fit(X_train, y_train, sample_weight=weights)
        y_pred = clf.predict(self.unlabel)
        if self.return_proba:
            y_prob = clf.predict_proba(self.unlabel)[:, -1]
            return y_pred, y_prob
        else:
            return y_pred
