# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def variance_contrast(X, contamination=0.01):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=None, random_state=2018)  
    pca.fit(X)
    variance_original = pca.explained_variance_
    
    # 运用孤立森林进行异常检测
    iforest = IsolationForest(contamination=contamination, random_state=2018, n_jobs=-1, behaviour="new")  
    anomaly_pred = iforest.fit_predict(X) 

    # 获取异常样本的索引，并删除异常样本
    anomaly_indices = np.argwhere(anomaly_pred==-1).ravel()
    X_revise = X[np.in1d(range(len(X)), anomaly_indices, invert=True)]
    
    # 对删除异常样本后的矩阵进行PCA
    pca.fit(X_revise)
    variance_revised = pca.explained_variance_
    
    # 对删除异常样本前后的特征值进行对比
    delta_ratio = (variance_revised - variance_original) / variance_original
    
    # 根据特征值减小程度的多少对索引降序排列
    # idx_desc_top3为特征值减小幅度最大的前3个索引
    idx_desc_top3 = np.argsort(delta_ratio)[:3]
    
    # min_max_idx为最小与最大特征值对应的索引，
    min_max_idx = [0,  X.shape[1]-1]
    # 如果 min_max_idx之中有任何一个索引出现在idx_desc_top3中
    # 则证明异常样本在最大以及最小的几个特征值对应的主成分上具有更大的分量
    # 或者说若最大以及最小的几个特征值对应的主成分构成的坐标轴不存在，则异常样本无法被线性表出
    result = any(np.in1d(min_max_idx, idx_desc_top3))
    return list(idx_desc_top3), result


def generate_dataset(seed, row=5000, col=20, contamination=0.02):
    rdg = np.random.RandomState(seed)
    outlier_num = int(row*contamination)
    inlier_num = row - outlier_num
    
    # 构造正常样本集，服从标准正态分布
    inliers = rdg.randn(inlier_num, col)
    
    # 构成异常样本集，由泊松分布、指数分布组合构成
    col_1 = col//2 if np.mod(col, 2) else int(col/2)
    col_2 = col - col_1
    outliers_sub_1 = rdg.poisson(1, (outlier_num, col_1))
    outliers_sub_2 = rdg.exponential(15, (outlier_num, col_2))
    outliers = np.c_[outliers_sub_1, outliers_sub_2]

    matrix = np.r_[inliers, outliers]
    return matrix

# 生成10个不重复的随机种子以及对应的数据集
seeds = np.random.choice(range(1000), size=10, replace=False)
matrices = list(map(generate_dataset, seeds))

# 输出验证结果
contrast_result = list(map(variance_contrast, matrices))
verify_result = pd.DataFrame(contrast_result, columns=['特征值降幅最大索引', '包含最大最小索引'])
print(verify_result)