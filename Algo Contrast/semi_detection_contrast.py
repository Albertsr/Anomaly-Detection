# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import time
import numpy as np  
import pandas as pd

from ADOA import ADOA
from pu_learning import PU_Learning as pul
from coverage import weighted_coverage
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *

# 函数generate_pudata用于生成适用于PU_Learning的数据集
# 参数seed为随机数种子，positive_size表示P集在整个数据集中的占比

def generate_pudata(seed, positive_size=0.25):
    rdg = np.random.RandomState(seed)  
    # row, col分别为数据集的行数与列数
    row = rdg.randint(8000, 10000)
    col = rdg.randint(30, 35)
    
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
    coverage = weighted_coverage(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    performance = [auc, coverage, f_score, gmean, recall, acc]
    return np.array(performance)

rf = RandomForestClassifier(n_estimators=350, max_depth=6, random_state=2018)
xgb = XGBClassifier(n_estimators=350, learning_rate=0.25, max_depth=6, n_jobs=-1, random_state=2018)

def performance_contrast(seed, Cminus=1, Cplus=1.5, clf_one=xgb, clf_two=rf):
    start = time.time()
    P, U, U_label = generate_pudata(seed)
    print('Seed:{:}, P Shape:{:}, U Shape:{:}'.format(seed, P.shape, U.shape))
    
    # ADOA
    adoa = ADOA(P, U, clf_one)
    y_pred, y_prob  = adoa.predict()
    adoa_performance = model_perfomance(y_pred, y_prob, U_label)

    pul_csl= pul(P, U, Cminus=Cminus, Cplus=Cplus, clf_one=clf_one, clf_two=clf_two, over_sample=False)
    y_pred, y_prob = pul_csl.predict()
    pul_csl_performance = model_perfomance(y_pred, y_prob, U_label)

    pul_oversampled = pul(P, U, Cminus=1, Cplus=1, clf_one=clf_one, clf_two=clf_two, over_sample=True)
    y_pred, y_prob = pul_oversampled.predict()
    pul_sampled_performance = model_perfomance(y_pred, y_prob, U_label)

    metrics = ['AUC', 'Coverage', 'F1_Score', 'G_Mean', 'Recall', 'ACC']
    models = ['ADOA', 'PUL OverSampled', 'PUL CostSensitive']
    list_ = [adoa_performance, pul_sampled_performance, pul_csl_performance]
    performance = pd.DataFrame(list_, columns=metrics, index=models)
    
    decription = 'The evaluation of the algorithm has been completed.'
    print(decription, 'Running_Time:{:.2f}s'.format(time.time()-start))
    return performance

# 对最高的分数予以标黄，仅适用于Jupyter
def highlight_max(s):
    is_max = s == s.max() 
    bg = ['background-color: yellow' if v else '' for v in is_max]
    return bg

performance_contrast(2017).style.apply(highlight_max, axis=0)

def return_algo(seed):
    contrast = performance_contrast(seed)
    algorithms = [contrast[i].idxmax() for i in contrast.columns]
    print(algorithms, '\n')
    return np.array(algorithms)

seeds = np.random.RandomState(2018).choice(range(1000), size=10, replace=False)
indices_sorted = list(map(return_algo, seeds))
index = ['Dataset_' + str(i) for i in range(len(seeds))]
algo_sorted = pd.DataFrame(indices_sorted, index=index)
algo_sorted.index.name = 'VerifyData'
sorted_algo = algo_sorted.copy()
mode = sorted_algo.mode(axis=0)

def revise_mode(mode):
    target_idx = mode.notnull().sum().idxmax()
    target_col = mode.iloc[:, target_idx]
    
    # 去掉first_row中在target_idx索引处的值，成为first_row_trimmed
    first_row = mode.iloc[0, :] 
    cond = np.isin(first_row.index, target_idx, invert=True)
    first_row_trimmed = first_row[cond]
    
    # target_col内元素不在first_row_trimmed之内，则保留，否则删除
    cond = np.isin(target_col, first_row_trimmed, invert=True)
    target_idx_mode = target_col[cond].values[0]
    
    first_row[target_idx] = target_idx_mode
    return first_row.values

if len(mode) == 1:
    sorted_algo.loc['Mode(众数)'] = mode.values.ravel()
else:
    sorted_algo.loc['Mode(众数)'] = revise_mode(mode)
    
metrics = ['AUC', 'Coverage', 'F1_Score', 'G_Mean', 'Recall', 'ACC']
sorted_algo.columns = metrics
print(sorted_algo)

# 对众数标黄，仅对Jupyter有效
def show(row):
    color = 'yellow'
    return 'background-color: %s' % color
sorted_algo.style.applymap(show, subset=pd.IndexSlice['Mode(众数)':, :])
