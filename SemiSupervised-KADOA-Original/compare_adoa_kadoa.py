# Author：MaXiao
# E-mail：maxiaoscut@aliyun.com


import numpy as np
import pandas as pd
from adoa import ADOA
from kadoa import KADOA
from collections import Counter
from lightgbm import LGBMClassifier
np.set_printoptions(precision=3, suppress=True)
pd.set_option('precision', 3)

from sklearn.metrics import *
def evaluate_model(y_true, y_pred, y_prob, index='model'):
    assert len(y_true) == len(y_pred)
    assert len(y_true) == len(y_prob)
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    gmean = np.sqrt(recall * precision)
    eval_frame = pd.DataFrame({'AUC':auc, 'F1':f1, 'G-Mean':gmean, 'ACC':acc, 
                               'Recall':recall, 'Precision':precision}, index=[index])
    return eval_frame

# 对高分值予以标黄，仅对Jupyter有效
def highlight_bg_max(s):
    is_max = s == s.max() # is_max是一个布尔型变量构成的矩阵
    bg_op = 'background-color: yellow'
    bg = [bg_op if v else '' for v in is_max]
    return bg

def generate_pudata(seed, anomaly_size=0.25):
    rdg = np.random.RandomState(seed)  
    # row, col分别为数据集的行数与列数
    row = rdg.randint(6000, 8000)
    col = rdg.randint(10, 15)
    
    # anomaly_num、unlabel_num分别为P集、U集包含的样本数
    anomaly_num = int(row * anomaly_size)
    unlabel_num = row - anomaly_num
    
    # contamination为U集中异常样本的占比
    # pos_u_num为U集中包含的正样本数
    contamination = rdg.uniform(0.1, 0.2)
    anomaly_unlabel_num = int(unlabel_num * contamination)
    
    # 异常样本同时分布于Unlabel set、Label set
    # 假设所有的异常样本分为3个簇，均不服从正态分布 
    anomaly_total_num = anomaly_num + anomaly_unlabel_num
    anomaly_sub_num = anomaly_total_num // 3
    
    anomalies_1 = rdg.uniform(-1.8, 1.8, size=(anomaly_sub_num, col))
    anomalies_2 = rdg.uniform(0, 1, size=(anomaly_sub_num, col)) 
    anomalies_3 = rdg.exponential(1.5, size=(anomaly_sub_num, col))
    anomalies_ = np.r_[anomalies_1, anomalies_2, anomalies_3]
    
    # anomalies_U是U集的子集
    rd_indices = rdg.permutation(len(anomalies_))
    anomalies_U = anomalies_[rd_indices[:anomaly_unlabel_num]]
    
    # 生成最终的P集，由观测到的anomalies构成
    P = anomalies_[rd_indices[anomaly_num:]]
    
    # 生成最终的无标签样本集，其中包含contamination比例的正样本
    # 假设正常样本服从标准正态分布
    U_neg = rdg.normal(loc=0, scale=1, size=(unlabel_num-anomaly_unlabel_num, col))  
    U = np.r_[U_neg, anomalies_U]
    U_label = np.r_[np.zeros(len(U_neg)), np.ones(len(anomalies_U))].astype(int)
    return P, U, U_label

seed = 2020
P, U, U_label = generate_pudata(seed)
P.shape, U.shape, Counter(U_label)

clf = LGBMClassifier(num_leaves=64, n_estimators=100)
a = KADOA(P, U, clf, kernel='linear', return_proba=True, verbose=2)
b = ADOA(P, U, clf, return_proba=True)

a_pred, a_prob = a.predict()
b_pred, b_prob = b.predict()
metrics_kadoa = evaluate_model(U_label, a_pred, a_prob, index='KADOA')
metrics_adoa = evaluate_model(U_label, b_pred, b_prob, index='ADOA')
metrics_contrast = pd.concat([metrics_adoa, metrics_kadoa], axis=0).round(3)
print(metrics_contrast)
