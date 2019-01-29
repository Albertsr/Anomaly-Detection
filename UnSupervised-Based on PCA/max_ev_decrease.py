# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# variance_contrast函数返回一个列表indices_desc_topk，以及一个布尔值bool_result
# indices_desc_topk包含k个元素，指k个降幅最大的特征值对应的索引
# bool_result用于记录indices_desc_topk中是否包含最小或最大的索引
def variance_contrast(X, k=3, contamination=0.01):
    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=None, random_state=2018)  
    pca.fit(X)
    # variance_original为各主成分对应的特征值，即样本在主成分空间内的投影方差
    variance_original = pca.explained_variance_
    
    # 运用孤立森林进行异常检测，得到异常样本索引anomaly_indices
    iforest = IsolationForest(contamination=contamination, random_state=2018, n_jobs=-1, behaviour="new")  
    anomaly_pred = iforest.fit_predict(X)
    anomaly_indices = np.argwhere(anomaly_pred==-1).ravel()
    
    # 删除异常样本，得到矩阵X_trimmed
    X_trimmed = X[np.isin(range(len(X)), anomaly_indices, invert=True)]
    # 对X_trimmed进行PCA，求得特征值variance_revised
    pca.fit(X_trimmed)
    variance_revised = pca.explained_variance_
    
    # 对删除异常样本前后的特征值进行对比
    delta_ratio = (variance_revised - variance_original) / variance_original
    # 只选取delta_ratio中的负数，确保对应特征值是下降的
    target_ratio = delta_ratio[delta_ratio < 0]
    # k为预设参数，表示选取特征值减小幅度最大的前k个索引
    if len(target_ratio) >= k: 
        indices_desc_topk = np.argsort(target_ratio)[:k]
    else:
        indices_desc_topk = np.argsort(target_ratio)[:len(target_ratio)]
    
    # min_max_idx为最小与最大特征值对应的索引
    min_max_idx = [0,  X.shape[1]-1] 
    # 验证min_max_idx之中是否有任何一个索引出现在indices_desc_topk中
    bool_result = any(np.isin(min_max_idx, indices_desc_topk))
    return indices_desc_topk, bool_result

# generate_dataset用于生成实验数据集
def generate_dataset(seed, row=5000, col=20, contamination=0.01):
    rdg = np.random.RandomState(seed)
    outlier_num = int(row*contamination)
    inlier_num = row - outlier_num
    
    # 构造服从标准正态分布的正常样本集
    inliers = rdg.randn(inlier_num, col)
    
    # 如果col为奇数，col_1=col//2，否则col_1=int(col/2)
    col_1 = col//2 if np.mod(col, 2) else int(col/2)
    col_2 = col - col_1
    # outliers_sub_1服从标准伽玛分布；outliers_sub_2服从指数分布
    # outliers_sub_1、outliers_sub_2在axis=1方向上予以整合，构成异常样本集
    outliers_sub_1 = rdg.standard_gamma(1, (outlier_num, col_1))
    outliers_sub_2 = rdg.exponential(5, (outlier_num, col_2))
    outliers = np.c_[outliers_sub_1, outliers_sub_2]
    
    # 将inliers与outliers在axis=0方向上予以整合，构成实验数据集
    matrix = np.r_[inliers, outliers]
    return matrix

# 生成10个不重复的随机种子以及对应的数据集
seeds = np.random.RandomState(2018).choice(range(100), size=10, replace=False)
matrices = list(map(generate_dataset, seeds))

# 输出验证结果
contrast_result = list(map(variance_contrast, matrices))
verify_result = pd.DataFrame(contrast_result, columns=['特征值降幅最大索引', '包含最大最小索引'])
print(verify_result)
