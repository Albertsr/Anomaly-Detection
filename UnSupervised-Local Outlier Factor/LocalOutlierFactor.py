# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np  
from sklearn.neighbors import LocalOutlierFactor


# 设置训练样本数及异常样本比例
n_samples = 1000  
contamination = 0.01    
n_inliers = int((1. - contamination) * n_samples)  
n_outliers = int(contamination * n_samples)  
  
rng = np.random.RandomState(2017)    
X = 0.3 * rng.randn(n_inliers // 2, 2)  

# 构建正常样本与异常样本  
inliers = np.r_[X + 2, X - 2]   
outliers = rng.uniform(low=-6, high=6, size=(n_outliers, 2))

# 正常样本与异常样本的融合  
X_train = np.r_[inliers, outliers] 

lof = LocalOutlierFactor(contamination=contamination, n_jobs=-1)
# fit_predict返回-1则表示为异常值；返回1表示非异常值
y_train_pred = lof.fit_predict(X_train)
# 返回异常样本的索引
outliers_indices = np.argwhere(y_train_pred==-1).ravel()
print('训练集异常样本索引 : {}'.format(outliers_indices))

# 属性negative_outlier_factor_：返回负的LOF score，其绝对值越大，样本越可能异常
lof_score = -lof.negative_outlier_factor_
outliers_indices_desc = np.argsort(-lof_score)[:len(outliers)]
print('按异常程度降序排列的训练集异常样本索引 : {}'.format(outliers_indices_desc))

# 生成测试集
n_samples = 1000  
contamination = 0.01    
n_inliers = int((1. - contamination) * n_samples)  
n_outliers = int(contamination * n_samples)  
  
rng = np.random.RandomState(2018)    
X = 0.3 * rng.randn(n_inliers // 2, 2)  

# 构建正常样本与异常样本  
inliers = np.r_[X + 2, X - 2]   
outliers = rng.uniform(low=-6, high=6, size=(n_outliers, 2))
X_test = np.r_[inliers, outliers] 

# 参数novelty必须设为True
lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, n_jobs=-1, novelty=True)
# 参数novelty=True时，clf.fit_predict(X_test)会报错
lof.fit(X_train)
y_test_pred = lof.predict(X_test)

# 返回异常样本的索引
outliers_indices = np.argwhere(y_test_pred==-1).ravel()
print('测试集异常样本索引 : {}'.format(outliers_indices))