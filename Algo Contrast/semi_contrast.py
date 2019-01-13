# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np  
import pandas as pd
from PUlearning import PUlearning 
from ADOA import ADOA
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *

def pu_dataset(row=10000, col=20, p_size=0.25, contamination=0.02, seed=2018):
    rdg = np.random.RandomState(seed)  
    p_num = int(np.ceil(row * p_size))
    u_num = row - p_num
    
    # pos_u_num为U集中包含的正样本数
    pos_u_num = int(u_num * contamination)
    row_sub = (p_num+pos_u_num) // 3

    # 将异常样本分为3个簇，分别服从卡方分布，标准伽马分布，指数分布
    anomalies_1 = rdg.chisquare(1, size=(row_sub, col))
    anomalies_2 = rdg.standard_gamma(3, size=(row_sub, col))
    anomalies_3 = rdg.exponential(5, size=(row_sub, col))
    anomalies_ = np.r_[anomalies_1, anomalies_2, anomalies_3]
    
    rd_indices = rdg.permutation(len(anomalies_))
    anomalies_U = anomalies_[rd_indices[:pos_u_num]]
    
    # 生成最终的正样本集，由观测到的anomalies构成
    P = anomalies_[rd_indices[pos_u_num:]]
    
    # 生成最终的无标签样本集，其中包含contamination比例的正样本
    U_neg = rdg.normal(loc=2, scale=1.5, size=(u_num-pos_u_num, col))  
    U = np.r_[U_neg, anomalies_U]
    U_label = np.r_[np.zeros(len(U_neg)), np.ones(len(anomalies_U))]
    return P, U, U_label

P, U, U_label = pu_dataset()
rf = RandomForestClassifier(n_estimators=350, max_depth=6, random_state=2018)
xgb = XGBClassifier(n_estimators=350, max_depth=6, learning_rate=0.15)

def adoa_result(model, P=P, U=U, clf=xgb):
    reliable_normal = model.classify()[1]
    potential_anomalies = model.classify()[2]
    reliable_normal_weight, potential_anomalies_weight = model.set_weight()
    X_train = np.r_[P, potential_anomalies, reliable_normal]

    # 生成y_train
    X_outliers_label = np.ones(P.shape[0])
    potential_anomalies_label = np.ones(potential_anomalies.shape[0])
    reliable_normal_label = np.zeros(reliable_normal.shape[0])
    y_train = np.r_[X_outliers_label, potential_anomalies_label, reliable_normal_label]
  
    # 生成权重
    weights = np.r_[np.ones(P.shape[0]), potential_anomalies_weight, reliable_normal_weight]

    clf.fit(X_train, y_train, sample_weight=weights)
    y_pred = clf.predict(U)
    y_prob = clf.predict_proba(U)[:, -1]
    return y_pred, y_prob

def model_perfomance(y_pred, y_prob, y_true=U_label):
    auc = roc_auc_score(y_true, y_prob)
    f_score = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    gmean = np.sqrt(recall * precision)
    acc = accuracy_score(y_true, y_pred)
    performance = [auc, f_score, gmean, recall, precision, acc]
    return np.array(performance)

y_pred, y_prob = adoa_result(model=ADOA(P, U, contamination=0.01))
res_adoa = model_perfomance(y_pred, y_prob)

pul_weighted = PUlearning(P, U, clf_one=xgb, clf_two=rf, over_sample=False)
y_pred, y_prob = pul_weighted.predict()
res_weighted = model_perfomance(y_pred, y_prob)

pul_reblanced = PUlearning(P, U, Cminus=1, Cplus=1, clf_one=xgb, clf_two=rf, over_sample=True)
y_pred, y_prob = pul_reblanced.predict()
res_reblanced = model_perfomance(y_pred, y_prob)

metrics = ['AUC', 'F_Score', 'G-Mean', 'Recall', 'Precision', 'ACC']
models = ['ADOA', 'PUL Weighted', 'PUL OverSampled']
model_performance = pd.DataFrame([res_adoa, res_weighted, res_reblanced], columns=metrics, index=models)
print(model_performance)

# 对最高的分数予以标黄，仅适用于Jupyter
def highlight_max(s):
    is_max = s == s.max() 
    bg = ['background-color: yellow' if v else '' for v in is_max]
    return bg

model_performance.style.apply(highlight_max, axis=0)