# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

from sklearn.svm import SVC
svc = SVC(C=1.0, kernel='rbf', gamma='auto', probability=True, random_state=2018)

def biased_svm(cost_fp, cost_fn, P, U, svm=svc, return_proba=True):
    assert cost_fn > cost_fp > 0, '对FN应赋予更高的代价' 
    
    X_train = np.r_[P, U]
    y_train = np.r_[np.ones(len(P)), np.zeros(len(U))]
    weight = [cost_fn if i else cost_fp for i in y_train]
    svm.fit(X_train, y_train, sample_weight=weight) 
    y_pred = svm.predict(U)
    if return_proba:
        y_prob = svm.predict_proba(U)[:, -1]
        return y_pred, y_prob
    else:
        return y_pred