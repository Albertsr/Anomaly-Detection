# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1200, random_state=2019, n_jobs=-1)

def weighted_lr(P, U, lr=lr, return_proba=True):
    X_train = np.r_[P, U]
    y_train = np.r_[np.ones(len(P)), np.zeros(len(U))]
    
    pos_weight = len(U) / len(X_train)
    neg_weight = 1 - pos_weight
    assert weight_pos > weight_neg > 0, '一般情况下，U集的个数应多于P集的个数' 
    
    weight = [pos_weight if i else neg_weight for i in y_train]
    lr.fit(X_train, y_train, sample_weight=weight)
    y_pred = lr.predict(U)
    
    if return_proba:
        y_prob = lr.predict_proba(U)[:, -1]
        return y_pred, y_prob
    else:
        return y_pred