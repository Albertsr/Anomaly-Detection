# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
import pandas as pd
from KADOA import KADOA
from ADOA import ADOA
from xgboost import XGBClassifier
from sklearn.metrics import *
xgb = XGBClassifier(n_estimators=350, learning_rate=0.15, max_depth=6, n_jobs=-1, random_state=2018)

# 函数generate_pudata用于生成适用于PU_Learning的数据集
# 参数seed为随机数种子，positive_size表示P集在整个数据集中的占比

def generate_pudata(seed, positive_size=0.25):
    rdg = np.random.RandomState(seed)  
    # row, col分别为数据集的行数与列数
    row = rdg.randint(2000, 3500)
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

def model_perfomance(y_pred, y_prob, y_true):
    auc = roc_auc_score(y_true, y_prob)
    f_score = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    gmean = np.sqrt(recall * precision)
    acc = accuracy_score(y_true, y_pred)
    performance = [auc, f_score, gmean, recall, acc]
    metrics = ['AUC', 'F1_Score', 'G_Mean', 'Recall', 'ACC']
    return pd.DataFrame(performance, index=metrics)

P, U, U_label = generate_pudata(2018)
adoa = ADOA(P, U, xgb)
kadoa = KADOA(P, U, xgb)
y_pred, y_prob = adoa.predict()
y_pred_k, y_prob_k = kadoa.predict()

adoa_result = model_perfomance(y_pred, y_prob, U_label)
kadoa_result = model_perfomance(y_pred_k, y_prob_k, U_label)

contrast = pd.concat([adoa_result.T, kadoa_result.T], axis=0)
contrast.index = ['ADOA', 'KADOA']
print(contrast)

# 对高分值予以标黄，仅对Jupyter有效
def highlight_bg_max(s):
    is_max = s == s.max() # is_max是一个布尔型变量构成的矩阵
    bg_op = 'background-color: yellow'
    bg = [bg_op if v else '' for v in is_max]
    return bg

contrast.style.apply(highlight_bg_max, axis=0)