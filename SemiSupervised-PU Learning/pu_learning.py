# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
from imblearn.over_sampling import SMOTE


class PUL:
    def __init__(self, P, U, clf_one, clf_two, cost_fn=1.5, cost_fp=1, return_proba=True, 
                 theta='auto', sample_ratio='auto', over_sample=False, random_state=2018):
        self.P = P  # P集由数据集中的正样本构成
        self.U = U  # U集由数据集中未标注的样本构成
        self.clf_one = clf_one  # step1所用的分类器，必须能返回取正的后验概率
        self.clf_two = clf_two  # step2所用的分类器
        self.cost_fn = cost_fn  # FN产生的代价
        self.cost_fp = cost_fp  # FP产生的代价
        self.return_proba = return_proba # 是否返回后验概率
        self.theta = theta      # 无标签样本判定为异常样本的阈值
        self.sample_ratio = 0.20 if sample_ratio=='auto' else sample_ratio # spy的抽样比例
        self.over_sample = over_sample  # 布尔型参数，表示是否在step2进行过采样 
        self.random_state = random_state
      
    def reliable_negative(self):        
        rdg = np.random.RandomState(self.random_state)
        # p_indices为P集随机排序后的索引
        p_indices = rdg.permutation(len(self.P))
        
        # spy_num为spy样本的数目，从P集中随机抽取spy_num个样本构成spy集
        spy_num = int(len(self.P) * self.sample_ratio)
        spy = self.P[p_indices[:spy_num]]
        
        # 将spy添加至U集，构成负样本集neg，对应标签为0 
        neg = np.r_[self.U, spy]
        neg_label = np.zeros(len(neg))
        
        # 原P集剔除spy后剩余的样本构成新的正样本集pos，标签为1
        pos = self.P[p_indices[spy_num:]]        
        pos_label = np.ones(len(pos))

        # 将pos、neg整合成训练集，用于训练clf_one       
        X_train = np.r_[pos, neg]
        y_train = np.r_[pos_label, neg_label]        
        clf1 = self.clf_one 
        clf1.fit(X_train, y_train)
        
        # 对U集、spy集进行预测，返回取正的后验概率
        y_prob_U = clf1.predict_proba(self.U)[:, -1]
        y_prob_spy = clf1.predict_proba(spy)[:, -1]
        
        # 确定阈值theta
        if self.theta == 'auto':
            # 设置theta为spy后验概率的最小值，使得spy内的所有样本全部准确预测为正样本
            theta = np.min(y_prob_spy)
        else:
            theta = self.theta
        
        # U集中取正的后验概率小于阈值theta的样本被判定为可靠负样本(reliable negatives)
        RN = self.U[y_prob_U < theta] 
        return RN, theta, np.min(y_prob_U), np.max(y_prob_U), np.mean(y_prob_U)
    
    def predict(self):
        # 对可靠负样本集的赋予标签0
        RN = self.reliable_negative()[0]    
        RN_label = np.zeros(len(RN))
        
        # P_label为原P集的标签
        P_label = np.ones(len(self.P))
        
        # 将RN、P整合成训练集，用于训练step2阶段的分类器clf_two
        X_train = np.r_[self.P, RN]
        y_train = np.r_[P_label, RN_label]
        pos_num = sum(y_train==1)
        neg_num = sum(y_train==0)
        
        # 根据需要决定是否进行过采样，当self.over_sample为True,且pos_num < neg_num时才进行过采样
        # 之所以要求pos_num < neg_num，是为了对正样本进行过采样，使得算法更充分地学习正样本，增强模型的Recall
        # 在风控场景中，FN的代价要显著高于FP的代价，因此应更注重提升模型的Recall，对Pecision适当关注即可
        if self.over_sample and pos_num<neg_num:
            #X_train, y_train = ADASYN(random_state=self.random_state).fit_resample(X_train, y_train)
            smote = SMOTE(kind='svm', random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        else:
            X_train, y_train = X_train, y_train
        
        # 设置样本权重
        weights = np.array([self.cost_fn if i else self.cost_fp for i in y_train])
        
        # 用P和RN训练分类器clf2，并对U进行预测
        clf2 = self.clf_two
        clf2.fit(X_train, y_train, sample_weight=weights)
        y_pred = clf2.predict(self.U)
        if self.return_proba:
            y_prob = clf2.predict_proba(self.U)[:, -1]
            return y_pred, y_prob
        else:    
            return y_pred 
