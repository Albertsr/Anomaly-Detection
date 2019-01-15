# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import time
import numpy as np
import pandas as pd

import mahal_dist as md
import Robust_PCC as rp
import Recon_Error_PCA as rep
import Recon_Error_KPCA as rek

from sklearn.datasets import *
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def predict_anomaly_indices(X, contamination):
    
    # 孤立森林
    iforest = IsolationForest(n_estimators=125, contamination=contamination, 
                              behaviour='new', random_state=2018, n_jobs=-1)
    iforest_result = iforest.fit_predict(X)
    anomaly_num = len(np.where(iforest_result==-1)[0])
    # 分数越小于0，越有可能是异常值
    anomaly_score = iforest.decision_function(X)
    if_idx = np.argsort(anomaly_score)[:anomaly_num]
    
    # LOF
    lof = LocalOutlierFactor(contamination=contamination, p=2, novelty=False, n_jobs=-1)
    lof.fit(X)
    score = -lof.negative_outlier_factor_ 
    lof_idx = np.argsort(-score)[:anomaly_num]
    
    # RobustPCC
    rpcc = rp.RobustPCC(X, X, gamma=0.01, quantile=99)
    rpcc_idx = rpcc.test_anomaly_idx()[:anomaly_num]
    
    # 马氏距离
    dist = md.mahal_dist(X)
    md_idx = np.argsort(-dist)[:anomaly_num]
    
    # LinearPCA重构
    pre = rep.PCA_Recon_Error(X, contamination=contamination)
    pre_idx = pre.anomaly_idx()
    
    # KernelPCA重构
    kre = rek.KPCA_Recon_Error(X, contamination=contamination)
    kre_idx = kre.anomaly_idx()
    
    anomaly_indices = [if_idx, lof_idx, rpcc_idx, md_idx, kre_idx, pre_idx]
    return np.array(anomaly_indices)

def anomaly_indices_contrast(X, contamination=0.02, observed_anomaly_indices=None):
    start = time.time()
    anomaly_indices = predict_anomaly_indices(X, contamination)
    if observed_anomaly_indices:
        baseline = observed_anomaly_indices  
    else: 
        # 如果异常样本的索引未知，则以孤立森林判定的异常索引为Baseline
        baseline = anomaly_indices[0]
    
    indices_contrast = pd.DataFrame(anomaly_indices)
    algorithms = ['Isolation Forest', 'LOF', 'Robust PCC', 'Mahalanobis Dist', 'KPCA_Recon_Error', 'PCA_Recon_Error']
    indices_contrast.index = algorithms
    indices_contrast.index.name = 'Algorithm'

    # 统计各算法预测出的异常索引与baseline的相交个数
    def indices_intersect(indices_predicted):
        return sum(np.isin(indices_predicted, baseline))

    # 在indices_contrast中新增一列'Baseline_Same'，用于存放各算法预测与baseline的相交个数
    indices_contrast['Baseline_Same'] = list(map(indices_intersect, anomaly_indices))
    # 根据Baseline_Same的取值大小，对indices_contrast各行进行降序排列
    indices_contrast.sort_values(by=['Baseline_Same'], ascending=False, inplace=True)
    print('Dataset_Shape:{:}, Running_Time:{:.2f}s'.format(X.shape, (time.time()-start)))
    return indices_contrast

boston = load_boston().data
print(anomaly_indices_contrast(boston))

cancer = load_breast_cancer().data
print(anomaly_indices_contrast(cancer))

def generate_dataset(seed):
    rdg = np.random.RandomState(seed)
    row = rdg.randint(2500, 3000)
    col = rdg.randint(30, 35)
    contamination = rdg.uniform(0.015, 0.025)
    
    outlier_num = int(row*contamination)
    inlier_num = row - outlier_num
    
    # 正常样本集服从标准正态分布
    inliers = rdg.randn(inlier_num, col)
    
    # 如果outlier_num为奇数，row_1=outlier_num//2，否则row_1=int(outlier_num/2)
    row_1 = outlier_num//2 if np.mod(outlier_num, 2) else int(outlier_num/2)
    row_2 = outlier_num - row_1
    
    # outliers_sub_1服从伽玛分布；outliers_sub_2服从指数分布
    outliers_sub_1 = rdg.gamma(shape=2, scale=0.5, size=(row_1 , col))
    outliers_sub_2 = rdg.exponential(1.5, size=(row_2, col))
    outliers = np.r_[outliers_sub_1, outliers_sub_2]
    
    # 将inliers与outliers在axis=0方向上予以整合，构成实验数据集
    dataset = np.r_[inliers, outliers]
    outliers_indices = range(len(dataset))[inlier_num:]
    return dataset, contamination, outliers_indices

def return_algo(seed):
    dataset, contamination, outliers_indices = generate_dataset(seed)
    result = anomaly_indices_contrast(dataset, contamination, outliers_indices)
    return result.index

seeds = np.random.RandomState(2018).choice(range(1000), size=10, replace=False)
indices_sorted = list(map(return_algo, seeds))
index = ['Dataset_' + str(i) for i in range(len(seeds))]
algo_sorted = pd.DataFrame(indices_sorted, index=index)
algo_sorted.index.name = 'VerifyData'
sorted_algo = algo_sorted.copy()
mode = sorted_algo.mode(axis=0)
print(mode)

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
    
#columns = ['Baseline','First Algorithm', 'Second Algorithm', 'Thrid Algorithm', 'Fourth Algorithm', 'Fifth Algorithm']
columns = ['Algorithm 1st', 'Algorithm 2nd', 'Algorithm 3rd', 'Algorithm 4th', 'Algorithm 5th', 'Algorithm 6th']
sorted_algo.columns = columns
print(sorted_algo.columns)

# 对众数进行标黄，仅对jupyter有效
def show(row):
    color = 'yellow'
    return 'background-color: %s' % color
sorted_algo.style.applymap(show, subset=pd.IndexSlice['Mode(众数)':, :])