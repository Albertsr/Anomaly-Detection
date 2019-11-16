# Author：Maxiao
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import time
import numpy as np
import pandas as pd
import seaborn as sns

import mahal_dist as md
import RobustPCC as rp
import PCA_Recon_Error as rep
import KPCA_Recon_Error as rek

from sklearn.metrics import *
from sklearn.datasets import *
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from matplotlib import pyplot as plt
from pandas.plotting import parallel_coordinates
%matplotlib inline

def predict_anomaly_indices(X, contamination):

    # 孤立森林
    iforest = IsolationForest(n_estimators=125, contamination=contamination, 
                              behaviour='new', random_state=2018, n_jobs=-1)
    # Returns -1 for outliers and 1 for inliers.
    iforest_pred = iforest.fit_predict(X)
    iforest_result = np.array([1 if pred==-1 else 0 for pred in iforest_pred])

    # LOF
    lof = LocalOutlierFactor(contamination=contamination, p=2, novelty=False, n_jobs=-1)
    # Returns -1 for outliers and 1 for inliers.
    lof_pred = lof.fit_predict(X)
    lof_result = np.array([1 if pred==-1 else 0 for pred in lof_pred])

    # 马氏距离
    dist = md.mahal_dist(X)
    anomaly_num = int(np.ceil(contamination * len(X)))
    md_idx = np.argsort(-dist)[:anomaly_num]
    mahal_result = np.array([1 if i in md_idx else 0 for i in range(len(X))])
       
    # RobustPCC
    rpcc = rp.RobustPCC(X, X, gamma=0.01, quantile=99, contamination=contamination)
    rpcc_result = rpcc.predict()  
         
    #LinearPCA重构
    pre = rep.PCA_Recon_Error(X, contamination=contamination)
    pre_result = pre.predict()
     
    ##KernelPCA重构
    kre = rek.KPCA_Recon_Error(X, contamination=contamination, kernel='linear')
    print('KernelPCA starts.')
    start = time.time()
    kre_result = kre.predict()
    end = time.time()
    print("KernelPCA cost time: {:.2f}s".format(end-start))
    
    anomaly_pred = [iforest_result, lof_result, mahal_result, pre_result, kre_result, rpcc_result]
    return np.array(anomaly_pred)


def evaluate_model(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    acc = accuracy_score(y_true, y_pred).round(4)
    f1 = f1_score(y_true, y_pred).round(4)
    recall = recall_score(y_true, y_pred).round(4)
    precision = precision_score(y_true, y_pred).round(4)
    decription = 'F1:{:.3f}, ACC:{:.3F}, Recall:{:.3f}, Precision:{:.3f}'
    df_temp = pd.DataFrame([f1, acc, recall, precision]).T
    df_temp.columns = ['F1', 'ACC', 'Recall', 'Precision']
    return df_temp


def contrast_models(X, y_true, metric=['f1']):
    contamination = sum(y_true) / len(X)
    anomaly_pred = predict_anomaly_indices(X, contamination)
    df_res = pd.concat([evaluate_model(y_true, i) for i in anomaly_pred])
    df_res.index = ['Isolation Forest', 'LOF', 'Mahalanobis Dist', 'PCA_Recon_Error', 'KPCA_Recon_Error', 'Robust PCC']
    cols1 = np.array(['f1', 'acc', 'recall', 'precision'])
    cols2 = np.array(['F1', 'ACC', 'Recall', 'Precision'])
    display_metrics = cols2[[np.argwhere(cols1==i)[0][0] for i in metric]]
    return pd.DataFrame(df_res.loc[:, display_metrics]).T


def generate_dataset(seed):
    rdg = np.random.RandomState(seed)
    row = rdg.randint(2500, 3000) #rdg.randint(2500, 3000)
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
    X = np.r_[inliers, outliers]
    y = np.r_[np.zeros(len(inliers)), np.ones(len(outliers))]
    return X, y

seeds = np.random.RandomState(2018).choice(range(1000), size=10, replace=False)
datasets = [generate_dataset(seed) for seed in seeds]


def get_metric_df(datasets, metric):
    df_metrics = pd.concat([contrast_models(i[0], i[1], metric=metric) for i in datasets])
    df_metrics['dataset'] = np.array([['Dataset_' + str(i)]*len(metric) for i in range(len(datasets))]).ravel()
    return df_metrics


def plot_parallel(df):
    plt.figure(figsize=(12, 6))
    plt.title(df.index[0]+' score of different algorithms', fontsize=15)
    parallel_coordinates(df, 'dataset') 
    plt.grid(lw=0.1)
    plt.legend(loc=4)
    plt.ylabel(df.index[0], fontsize=14)
    plt.show()
    
df_metrics = get_metric_df(datasets, ['f1', 'acc', 'recall', 'precision'])    
plot_parallel(df_metrics.loc['F1'])
