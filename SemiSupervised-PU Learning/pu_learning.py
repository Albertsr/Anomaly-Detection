# Author：MaXiao
# E-mail：maxiaoscut@aliyun.coms
 
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression

class PULearning:
    """Implementation of PULearning (two-step strategy & cost-sensitive strategy)"""
    def __init__(self, P, U, clf_one, clf_two, Cplus, Cminus=1, sample_ratio=0.15, 
                 theta='auto', random_state=2018):
        """
        :param P: Observed positive samples.
        :param U: Unlabeled datasets.
        :param clf_one: A Classifer used to determine reliable negative samples must be able to predict probability.
        :param clf_two: A Classifer fit positive samples and reliable negative samples, and predict the unlabeled.
        :param Cplus: the cost of not identifying positive samples, cost(FN) 
        :param Cminus: the cost of not identifying negative samples, cost(FP) 
        :param sample_ratio: the proportion of spy samples
        :param theta: the probability threshold of judging an unlabeled sample as a reliable negative sample
        """
        self.P = P  
        self.U = U   
        assert clf_one.predict_proba, 'need predict_proba method to return probability estimates'
        self.clf_one = clf_one  
        self.clf_two = clf_two 
        self.Cplus = Cplus
        self.Cminus = Cminus
        self.theta = theta
        self.sample_ratio = 0.15 if sample_ratio=='auto' else sample_ratio
        self.random_state = random_state
        
     
    # Two-Stage Strategy: Select Reliable Negative Instances
    def select_reliable_negative(self): 
        pos_num = len(self.P)
        spy_num = int(pos_num * self.sample_ratio)
        pos_random_indices = np.random.RandomState(self.random_state).permutation(pos_num)
        spy_indices, unspy_indices = pos_random_indices[:spy_num], pos_random_indices[spy_num:]
        spy_set, unspy_set = self.P[spy_indices, :], self.P[unspy_indices, :]
        
        negative_set = np.r_[self.U, spy_set]
        positive_set = unspy_set 
        negative_label = np.zeros(len(negative_set)).astype(int)   
        positive_label = np.ones(len(positive_set)).astype(int)
     
        X_train_one = np.r_[negative_set, positive_set]
        y_train_one = np.r_[negative_label, positive_label].astype(int)    
        clf_one = self.clf_one.fit(X_train_one, y_train_one)
        
        y_prob_U = clf_one.predict_proba(self.U)[:, 1]
        y_prob_spy = clf_one.predict_proba(spy_set)[:, 1]
        
        theta = np.min(y_prob_spy) if self.theta == 'auto' else self.theta
        assertion = 'theta must not be greater than the minimum value of spy_prob so that \
            all spy are predicted to be positive samples'
        assert theta <= np.min(y_prob_spy), assertion
        
        # rn: reliable_negative
        rn = self.U[y_prob_U <= theta, :] 
        return rn
    
    def predict(self):
        # 对可靠负样本集的赋予标签0
        rn = self.select_reliable_negative()    
        X_train_two = np.r_[self.P, rn]
        y_train_two = np.r_[np.ones(len(self.P)), np.zeros(len(rn))].astype(int)
        weights = np.array([self.Cplus if i else self.Cminus for i in y_train_two])
        
        clf_two = self.clf_two
        clf_two.fit(X_train_two, y_train_two, sample_weight=weights)
        y_pred = clf_two.predict(self.U)
        
        if clf_two.predict_proba:
            y_prob = clf_two.predict_proba(self.U)[:, -1]
        return y_pred, y_prob
