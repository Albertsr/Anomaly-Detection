# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr


import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE


class PULearning:
    def __init__(self, P, U, clf_one, clf_two, Cplus=1.5, Cminus=1, theta='auto', 
                 sample_ratio='auto', over_sample=False, random_state=2018):
        self.P = P  # P集由数据集中的正样本构成
        self.U = U  # U集由数据集中未标注的样本构成
        self.clf_one = clf_one  # step1所用的分类器，必须能返回取正的后验概率
        self.clf_two = clf_two  # step2所用的分类器
        self.Cplus = Cplus
        self.Cminus = Cminus
        self.theta = theta
        self.sample_ratio = 0.25 if sample_ratio=='auto' else sample_ratio
        self.over_sample = over_sample  
        self.random_state = random_state
      
    def get_rn(self):        
        rdg = np.random.RandomState(self.random_state)
        # p_indices为P集随机排序后的索引
        p_indices = rdg.permutation(self.P.shape[0])
        
        # spy_num为spy样本的数目，从P集中随机抽取spy_num个样本构成spy集
        spy_num = int(len(self.P) * self.sample_ratio)
        spy = self.P[p_indices[:spy_num]]
        
        # 将spy添加至U集，构成负样本集neg，对应标签为0 
        neg = np.r_[self.U, spy]
        neg_label = np.zeros(len(neg))
        
        # 原P集剔除spy后剩余的样本构成新的正样本集pos，标签为1
        pos = self.P[p_indices[spy_num:]]        
        pos_label = np.ones(len(pos))

        # 将pos、neg进行整合，用于训练clf_one       
        X_train = np.r_[pos, neg]
        y_train = np.r_[pos_label, neg_label]        
        clf_one = self.clf_one 
        clf_one.fit(X_train, y_train)
        
        # 对U集、spy集进行预测，返回取正的后验概率
        y_prob_U = clf_one.predict_proba(self.U)[:, -1]
        y_prob_spy = clf_one.predict_proba(spy)[:, -1]
        
        # 确定阈值theta
        if self.theta == 'auto':
            # 设置theta为最小值，使得spy内的所有样本全部准确预测为正样本
            theta = np.min(y_prob_spy)
        else:
            theta = self.theta
        
        # print('theta: {:}'.format(theta))
        # U集中取正的后验概率小于阈值theta的样本被判定为可靠负样本(reliable negatives)
        RN = self.U[y_prob_U <= theta] 
        return RN, theta, np.min(y_prob_U), np.max(y_prob_U), np.mean(y_prob_U)
    
    def predict(self):
        # 对可靠负样本集的赋予标签0
        RN = self.get_rn()[0]    
        RN_label = np.zeros(len(RN))
        
        # P_label为原P集的标签
        P_label = np.ones(len(self.P))
        
        # 对RN、P予以整合，训练clf_two
        X_train = np.r_[self.P, RN]
        y_train = np.r_[P_label, RN_label]
        pos_num = sum(y_train==1)
        neg_num = sum(y_train==0)
        
        # 过采样
        if self.over_sample and pos_num<neg_num:
            # X_train, y_train = ADASYN(random_state=self.random_state).fit_resample(X_train, y_train)
            smote = SMOTE(kind='svm', random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        else:
            X_train, y_train = X_train, y_train

        # 获取过采样操作后正负样本的个数
        pos_num = sum(y_train==1)
        neg_num = sum(y_train==0)
        print(neg_num, pos_num)
        
        # 设置样本权重
        weights = np.array([self.Cplus if i else self.Cminus for i in y_train])
        
        # 用P和RN训练分类器clf2，并对U进行预测
        clf_two = self.clf_two
        clf_two.fit(X_train, y_train, sample_weight=weights)
        y_pred = clf_two.predict(self.U)
        y_prob = clf_two.predict_proba(self.U)[:, -1]
        return y_pred, y_prob