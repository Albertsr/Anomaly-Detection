# Author：MaXiao
# E-mail：maxiaoscut@aliyun.com

import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


# compare_variance: used to observe the influence of eliminating abnormal sample on the corresponding eigenvalues of PCA.

def compare_variance(X, k=3, contamination=0.01):
    """
    :param X: {array-like, sparse matrix, dataframe} of shape (n_samples, n_features)
    :param k: int, the number of retained eigenvalues
    :param contanination: float, range (0, 0.5), The proportion of outliers in the data set.  
    """
    X = X.values if isinstance(X, pd.DataFrame) else np.array(X)
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=None, random_state=2018)  
    pca.fit(X_scaled)
    variance_original = pca.explained_variance_
    
    # IsolationForest is used for anomaly detection, and the anomaly_indices is obtained.
    iforest = IsolationForest(contamination=contamination, random_state=2018, n_jobs=-1)  
    anomaly_label = iforest.fit_predict(X_scaled)
    anomaly_indices = np.argwhere(anomaly_label==-1).ravel()
    
    # delete the exception sample and get the matrix X_filter
    X_filter = X_scaled[np.isin(range(len(X_scaled)), anomaly_indices, invert=True), :]
    pca.fit(X_filter)
    variance_filter = pca.explained_variance_
    
    # compare the eigenvalues before and after deleting the abnormal sample.
    # Only negative numbers in delta_ratio are selected to ensure that the corresponding eigenvalues are reduced.
    delta_ratio = (variance_filter - variance_original) / variance_original
    target_ratio = delta_ratio[delta_ratio < 0]
    
    # select the index with the largest decrease in eigenvalues
    if len(target_ratio) >= k: 
        indices_topk = np.argsort(target_ratio)[:k]
    else:
        indices_topk = np.argsort(target_ratio)[:len(target_ratio)]
    
    # verify that any one of the maximum or minimum index appears in the indices_topk
    indices_min_max = [0,  X.shape[1]-1] 
    bool_result = any(np.isin(indices_min_max, indices_topk))
    return indices_topk, bool_result


# generate_dataset用于生成实验数据集
def generate_dataset(seed, row=5000, col=20, contamination=0.01):
    rdg = np.random.RandomState(seed)
    outlier_num = int(row*contamination)
    inlier_num = row - outlier_num
    
    # construct a normal sample set that obeys standard normal distribution
    inliers = rdg.randn(inlier_num, col)
    
    # If col is odd，col_1=col//2，else col_1=int(col/2)
    col_1 = col//2 if np.mod(col, 2) else int(col/2)
    col_2 = col - col_1
    
    # outliers_sub_1 obeys standard gamma distribution 
    # outliers_sub_2 obeys exponential distribution.
    outliers_sub_1 = rdg.standard_gamma(1, (outlier_num, col_1))
    outliers_sub_2 = rdg.exponential(5, (outlier_num, col_2))
    outliers = np.c_[outliers_sub_1, outliers_sub_2]
    
    matrix = np.r_[inliers, outliers]
    return matrix

# generate 10 non-repetitive random seeds and corresponding data sets
seeds = np.random.RandomState(2018).choice(range(100), size=10, replace=False)
matrices = [generate_dataset(seed) for seed in seeds]

# output verification results
contrast_result = [compare_variance(matrix) for matrix in matrices]
verify_result = pd.DataFrame(contrast_result, columns=['target_index', 'contain_minmax'])
print(verify_result)
