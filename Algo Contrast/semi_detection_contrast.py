# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import time
import numpy as np  
import pandas as pd

from ADOA import ADOA
from pu_learning import PUL as pul
from coverage import weighted_coverage
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.svm import SVC


# 函数generate_pudata用于生成适用于PU_Learning的数据集
# 参数seed为随机数种子，positive_size表示P集在整个数据集中的占比

def generate_pudata(seed, positive_size=0.25):
    rdg = np.random.RandomState(seed)  
    # row, col分别为数据集的行数与列数
    row = rdg.randint(6000, 8000)
    col = rdg.randint(45, 55)
    
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
    
    col_1 = int(col * 0.65)
    col_2 = col - col_1
    a = rdg.uniform(-1, 0, size=(row_sub, col_1))
    b = rdg.rayleigh(1, size=(row_sub, col_2))
    anomalies_1 = np.c_[a, b]
    anomalies_2 = rdg.uniform(0, 1, size=(row_sub, col))
    anomalies_3 = rdg.exponential(2.5, size=(row_sub, col)) #1.5 #2 ADOA最佳
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
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = tp / (tp+fn)
    specificity = tn / (tn+fp)
    gmean = np.sqrt(recall * specificity)
    
    auc = roc_auc_score(y_true, y_prob)
    f_score = f1_score(y_true, y_pred)
    coverage = weighted_coverage(y_true, y_prob)
    acc = accuracy_score(y_true, y_pred)
    performance = [auc, coverage, f_score, gmean, recall, acc]
    return np.array(performance)


rf = RandomForestClassifier(n_estimators=350, max_depth=6, random_state=2018)
xgb = XGBClassifier(n_estimators=350, learning_rate=0.25, max_depth=6, n_jobs=-1, random_state=2018)
svm = SVC(C=1.0, kernel='rbf', gamma='auto', probability=True, degree=3, random_state=2018)
lr = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1200, random_state=2019, n_jobs=-1)

def performance_contrast(seed, cost_fn=1.5, cost_fp=1.0, clf_one=xgb, clf_two=rf):
    start = time.time()
    P, U, U_label = generate_pudata(seed)
    print('Seed:{:}, P Shape:{:}, U Shape:{:}'.format(seed, P.shape, U.shape))
    
    # ADOA
    adoa = ADOA(P, U, clf_one)
    y_pred, y_prob  = adoa.predict()
    adoa_performance = model_perfomance(y_pred, y_prob, U_label)
    
    # BiasedSVM
    X_train = np.r_[P, U]
    y_train = np.r_[np.ones(len(P)), np.zeros(len(U))]
    svm.fit(X_train, y_train, sample_weight=[cost_fn if i else cost_fp for i in y_train]) 
    y_pred, y_prob  = svm.predict(U), svm.predict_proba(U)[:, -1]
    svm_performance = model_perfomance(y_pred, y_prob, U_label)
    
    # Weighted LR
    pos_weight = len(U) / len(X_train)
    neg_weight = 1 - pos_weight
    lr.fit(X_train, y_train, sample_weight=[pos_weight if i else neg_weight for i in y_train])
    y_pred, y_prob  = lr.predict(U), lr.predict_proba(U)[:, -1]
    lr_performance = model_perfomance(y_pred, y_prob, U_label)
    
    # PUL CostSensitive
    pul_csl= pul(P, U, cost_fn=cost_fn, cost_fp=cost_fp, clf_one=clf_one, clf_two=clf_two, over_sample=False)
    y_pred, y_prob = pul_csl.predict()
    pul_csl_performance = model_perfomance(y_pred, y_prob, U_label)

    #pul_oversampled = pul(P, U, cost_fn=1, cost_fp=1, clf_one=clf_one, clf_two=clf_two, over_sample=True)
    #y_pred, y_prob = pul_oversampled.predict()
    #pul_sampled_performance = model_perfomance(y_pred, y_prob, U_label)

    metrics = ['AUC', 'Coverage', 'F1_Score', 'G_Mean', 'Recall', 'ACC']
    models = ['ADOA', 'Biased_SVM', 'Weighted_LR', 'PUL_CostSensitive']
    list_ = [adoa_performance, svm_performance, lr_performance, pul_csl_performance]
    performance = pd.DataFrame(list_, columns=metrics, index=models)
    
    algorithms = [performance[i].idxmax() for i in metrics]
    performance.loc['The Best Algorithm', :] = algorithms
    print(algorithms)
    
    decription = 'The evaluation of the algorithm has been completed.'
    print(decription, 'Running_Time:{:.2f}s\n'.format(time.time()-start))
    return performance


seeds = np.random.RandomState(2019).choice(range(1000), size=10, replace=False)
contrast = [performance_contrast(seed) for seed in seeds]
contrast_concat = np.concatenate([contrast[i] for i in range(len(contrast))])

data_names = np.array([['Dataset_' + str(i)]*5 for i in range(len(seeds))]).ravel()
models = ['ADOA', 'Biased_SVM', 'Weighted_LR', 'PUL_CostSensitive', 'The Best Algorithm'] * len(seeds) # CostSensitive
metrics = ['AUC', 'Coverage', 'F1_Score', 'G_Mean', 'Recall', 'ACC']
arrays = [data_names, models]
idx = pd.MultiIndex.from_arrays(arrays, names=('VerifyData', 'Algorithm'))
contrast_result = pd.DataFrame(contrast_concat, index=idx, columns=metrics)
print(contrast_result)

best_algo = contrast_result.query("Algorithm == 'The Best Algorithm'")
def real_mode(df):
    mode = df.mode(axis=0)
    if len(mode) == 1:
        return mode.values.ravel()
    else:
        target_idx = np.where(df.columns==mode.notnull().sum().idxmax())[0][0]
        target_name = df.columns[target_idx]
        target_col = mode.iloc[:, target_idx]        
        
        target_df = contrast_result[target_name].swaplevel()
        mean_value = [target_df[model].mean() for model in target_col]
            
        idx_max = np.argmax(mean_value)
  
    # 去掉first_row中在target_idx索引处的值，成为first_row_trimmed
    first_row = mode.iloc[0, :] 
    cond = np.isin(first_row.index, target_idx, invert=True)
    first_row_trimmed = first_row[cond]
    
    target_idx_mode = target_col[idx_max]
    first_row[target_idx] = target_idx_mode
    return first_row.values

algo_best = best_algo.copy() 
algo_best.loc[('All Datesets', 'Algorithm Mode(众数)'), :] = real_mode(algo_best)
print(algo_best)


# 对众数标黄，仅对Jupyter有效
def show(row):
    color = 'yellow'
    return 'background-color: %s' % color

algo_best.style.applymap(show, subset=pd.IndexSlice[('All Datesets', 'Mode(众数)'):, :])
