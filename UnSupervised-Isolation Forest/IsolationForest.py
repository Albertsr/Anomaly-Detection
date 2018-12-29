# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np  
import pandas as pd
from sklearn.ensemble import IsolationForest  


'''
API简要说明：
sklearn.ensemble.IsolationForest(n_estimators=100, max_samples='auto', contamination='legacy', max_features=1.0, 
                                 bootstrap=False, n_jobs=None, behaviour=’old’, random_state=None, verbose=0)
n_estimators：iTree的个数；
max_samples：构建单颗iTree的样本数；
contamination：异常值的比例；
max_features：构建单颗iTree的特征数；
bootstrap：布尔型参数，默认取False，表示构建iTree时有放回地进行抽样；
'''

# 设置训练样本数及异常样本比例
n_samples = 10000  
outliers_fraction = 0.25    
n_inliers = int((1. - outliers_fraction) * n_samples)  
n_outliers = int(outliers_fraction * n_samples)  
  
# //表示整数除法  
rng = np.random.RandomState(123)    
X = 0.3 * rng.randn(n_inliers // 2, 2)  

# 构建正常样本与异常样本  
X_train = np.r_[X + 2, X - 2]   
outliers = rng.uniform(low=-6, high=6, size=(n_outliers, 2))

# 正常样本与异常样本的融合  
X_train = np.r_[X_train, outliers]  

clf = IsolationForest(contamination=outliers_fraction, random_state=2018, n_jobs=-1, behaviour="new")  
# predict / fit_predict方法返回每个样本是否为正常值，若返回1表示正常值，返回-1表示异常值
y_pred_train = clf.fit_predict(X_train)  
pred = np.array(['正常' if i==1 else '异常' for i in y_pred_train])

# 分数越小于0，越有可能是异常值
scores_pred = clf.decision_function(X_train) 
dict_ = {'anomaly_score':scores_pred, 'y_pred':y_pred_train, 'result':pred}
scores = pd.DataFrame(dict_)
print(scores.sample(5))
