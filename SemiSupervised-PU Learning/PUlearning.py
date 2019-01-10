# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr


import numpy as np
from xgboost import XGBClassifier
from imblearn.over_sampling import ADASYN
from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier(n_estimators=350, random_state=2018)
xgb = XGBClassifier(n_estimators=350, learning_rate=0.15)

class PUlearning:
    def __init__(self, P, U, Cplus=1.5, Cminus=1, theta='auto', sample_ratio='auto', 
                 over_sample=False, clf1=rf, clf2=xgb, random_state=2018):
        self.P = P
        self.U = U
        self.Cplus = Cplus
        self.Cminus = Cminus
        self.clf1 = clf1
        self.clf2 = clf2
        self.theta = theta
        self.s = sample_ratio
        self.over_sample = over_sample  
        self.random_state = random_state
      
    def reliable_negative_set(self):
        if self.s == 'auto':
            s = 0.2
        else:
            s = self.s
        
        S_num = int(len(self.P) * s)
        rdg = np.random.RandomState(self.random_state)
        rp = rdg.permutation(self.P.shape[0])
        S_idx = rp[:S_num]
        Ps_idx = rp[S_num:]
        
        S = self.P[S_idx]
        Ps = self.P[Ps_idx]
        Us = np.r_[self.U, S]
        Ps_label = np.ones(len(Ps))
        Us_label = np.zeros(len(Us))
              
        X_train = np.r_[Ps, Us]
        y_train = np.r_[Ps_label, Us_label]
        
        clf1 = self.clf1
        clf1.fit(X_train, y_train)
        # 对S进行预测，确定阈值theta
        y_pred_prob_S = clf1.predict_proba(S)[:, -1]
        
        if self.theta == 'auto':
            # 设置theta为最小值，使得S内的样本全部准确预测为正样本
            theta = np.min(y_pred_prob_S)
        else:
            self.theta == theta
            
        y_prob_U = clf1.predict_proba(self.U)[:, -1]
        RN = self.U[y_prob_U <= theta] 
        return RN, theta, np.min(y_prob_U), np.max(y_prob_U), np.mean(y_prob_U)
    
    def predict(self):
        RN = self.reliable_negative_set()[0]    
        P_label = np.ones(len(self.P))
        RN_label = np.zeros(len(RN))
        X_train = np.r_[self.P, RN]
        y_train = np.r_[P_label, RN_label]

        # RN的样本数一般多于P的样本数，过采样操作有助于增加正样本数，提升模型的Recall
        if self.over_sample:
            X_train_real, y_train_real = ADASYN(random_state=self.random_state).fit_sample(X_train, y_train)
        else:
            X_train_real, y_train_real = X_train, y_train

        # 获取过采样操作后正负样本的个数
        P_real_num = len(y_train_real[y_train_real==1])
        RN_real_num = len(y_train_real[y_train_real==0])

        # 设置样本权重
        P_weight = np.ones(P_real_num) * self.Cplus
        RN_weight = np.ones(RN_real_num) * self.Cminus
        weight = np.r_[P_weight, RN_weight]
        
        # 用P和RN训练分类器clf2，并对U进行预测
        clf2 = self.clf2
        clf2.fit(X_train_real, y_train_real, sample_weight=weight)
        y_pred = clf2.predict(self.U)
        y_prob = clf2.predict_proba(self.U)[:, -1]
        return y_pred, y_prob