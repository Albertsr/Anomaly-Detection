# Author：MaXiao
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
import pandas as pd
from adoa import ADOA
from kadoa import KADOA
from lightgbm import LGBMClassifier
np.set_printoptions(precision=3, suppress=True)
pd.set_option('precision', 3)


def generate_pudata(seed, positive_size=0.3):
    rdg = np.random.RandomState(seed)  
    # row, col分别为数据集的行数与列数
    row = rdg.randint(10000, 15000)
    col = rdg.randint(10, 15)
    
    # contamination为U集中正样本的占比
    contamination = rdg.uniform(0.025, 0.035)
    
    # p_num、u_num分别为P集、U集包含的样本数
    p_num = int(np.ceil(row * positive_size))
    u_num = row - p_num
    
    # pos_u_num为U集中包含的正样本数
    pos_u_num = int(np.ceil(u_num * contamination))
    # 将异常样本分为3个簇，分别服从卡方分布，标准伽马分布，指数分布
    pos_num = p_num + pos_u_num
    row_sub = pos_num // 3
    
    anomalies_1 = rdg.uniform(-1.8, 1.8, size=(row_sub, col))
    anomalies_2 = rdg.uniform(0, 1, size=(row_sub, col)) 
    anomalies_3 = rdg.exponential(1.5, size=(row_sub, col))
    anomalies_ = np.r_[anomalies_1, anomalies_2, anomalies_3]
    
    rd_indices = rdg.permutation(len(anomalies_))
    anomalies_U = anomalies_[rd_indices[:pos_u_num]]
    
    # 生成最终的正样本集，由观测到的anomalies构成
    P = anomalies_[rd_indices[pos_u_num:]]
    
    # 生成最终的无标签样本集，其中包含contamination比例的正样本
    U_neg = rdg.rand(u_num-pos_u_num, col)  
    U = np.r_[U_neg, anomalies_U]
    U_label = np.r_[np.zeros(len(U_neg)), np.ones(len(anomalies_U))]
    return P, U, U_label

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
    eval_frame = pd.DataFrame({'AUC':auc, 'F1':f1, 'G-Mean':gmean,
                               'ACC':acc, 'Recall':recall, 'Precision':precision}, index=[index])
    return eval_frame

seed = 2020
P, U, U_label = generate_pudata(seed)
clf = LGBMClassifier(num_leaves=64, n_estimators=100, random_state=seed)

a = KADOA(P, U, clf, kernel='rbf', return_proba=True, verbose=2)
b = ADOA(P, U, clf, return_proba=True)

a_pred, a_prob = a.predict()
b_pred, b_prob = b.predict()

metrics_kadoa = evaluate_model(U_label, a_pred, a_prob, index='KADOA')
metrics_adoa = evaluate_model(U_label, b_pred, b_prob, index='ADOA')
metrics_contrast = pd.concat([metrics_adoa, metrics_kadoa], axis=0) 

def highlight_bg_max(s):
    is_max = s == s.max() # is_max是一个布尔型变量构成的矩阵
    bg_op = 'background-color: yellow'
    bg = [bg_op if v else '' for v in is_max]
    return bg

metrics_contrast.style.apply(highlight_bg_max, axis=0)
