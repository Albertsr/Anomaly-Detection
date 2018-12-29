# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import itertools
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, Birch
from sklearn.metrics import calinski_harabaz_score


def get_cluster_centers(unlabeled, clusters='auto', cluster_algo='kmeans', param_grid='auto', random_state=2018):
    
    # 既可以指定单个整数形式的簇数n_clusters，也可以设置一个取值范围
    clusters_cond = isinstance(clusters, int) and clusters>=2
    clusters_range = [clusters] if clusters_cond else range(2, 6)
    
    # get_centers函数可根据聚类算法对无标签数据集的预测结果返回各聚类簇的中心
    def get_centers(y_pred):
        centers = []
        for i in np.unique(y_pred):
            subset = unlabeled[y_pred==i]
            center = np.mean(subset, axis=0)
            centers.append(center)
        return np.sort(centers, axis=0)
    
    # 通过生成参数的笛卡尔积，寻求谱聚类算法的最优参数
    if cluster_algo == 'spectral':
        if param_grid == 'auto': 
            param_grid = {'n_clusters':clusters_range, 'gamma':np.linspace(0.5, 1.5, 3)}
        params, score, y_pred_set = [], [], []
        for i, j in itertools.product(param_grid['n_clusters'], param_grid['gamma']):
            params.append((i, j))
            spectral = SpectralClustering(n_clusters=i, gamma=j, n_jobs=-1, random_state=random_state)
            y_pred_spectral = spectral.fit_predict(unlabeled)
            y_pred_set.append(y_pred_spectral)
            ch_score = calinski_harabaz_score(unlabeled, y_pred_spectral)
            score.append(ch_score)
        # 获取calinski_harabaz_score取最大值时对应的参数与预测聚类类标
        best_param = params[np.argmax(score)]
        y_pred = y_pred_set[np.argmax(score)]            
        return get_centers(y_pred)
    
    # 通过生成参数的笛卡尔积，寻求Birch聚类算法的最优参数
    elif cluster_algo == 'birch':
        if param_grid == 'auto':
            param_grid = {'n_clusters':clusters_range, 'branching_factor':range(2,10), 'threshold':np.linspace(0, 0.8, num=10)}
            
        params, score, y_pred_set = [], [], []
        for i, j, k in itertools.product(param_grid['n_clusters'], param_grid['branching_factor'], param_grid['threshold']):
            params.append((i, j, k))
            birch = Birch(n_clusters=i, branching_factor=j, threshold=k)
            y_pred_birch = birch.fit_predict(unlabeled)
            y_pred_set.append(y_pred_birch)
            ch_score = calinski_harabaz_score(unlabeled, y_pred_birch)
            score.append(ch_score)
        best_param = params[np.argmax(score)]
        y_pred = y_pred_set[np.argmax(score)]
        return get_centers(y_pred)
    
    
    # 通过生成参数的笛卡尔积，寻求DBSCAN聚类算法的最优参数
    elif cluster_algo == 'dbscan':
        if param_grid == 'auto':
            param_grid = {'eps':np.linspace(0.1, 10, num=50), 'min_samples':range(1, 10)}
            
        params, unlabeled_set, y_pred_set, score = [], [], [], []
        for i, j in itertools.product(param_grid['eps'], param_grid['min_samples']):
            dbscan = DBSCAN(eps=i, min_samples=j, n_jobs=-1)
            y_pred_dbscan = dbscan.fit_predict(unlabeled)
            
            # DBSCAN视预测结果为-1的样本为噪声，因此需要将“噪音样本”予以排除
            y_pred_new = y_pred_dbscan[y_pred_dbscan != -1]
            unlabeled_new = unlabeled[y_pred_dbscan != -1]
            
            # 计算剔除“噪音样本”后无标签样本的剩余比例以及聚类簇的数目
            ratio = unlabeled_new.shape[0] / unlabeled.shape[0]
            n_clusters = len(np.unique(y_pred_new))
            
            # 剩余样本的聚类簇数以及剩余比例满足一定要求，才能对参数及预测结果予以保留
            if n_clusters in range(2, 6) and ratio>=0.8:
                params.append(i, j)
                unlabeled_set.append(unlabeled_new)
                y_pred_set.append(y_pred_new)
                ch_score = calinski_harabaz_score(unlabeled_new, y_pred_new)
                score.append(ch_score)
                
        best_param = params[np.argmax(score)]
        unlabeled_final = unlabeled_set[np.argmax(score)]
        y_pred = y_pred_set[np.argmax(score)]
        return get_centers(y_pred)
    
    # 寻求Kmeans聚类算法的最优参数
    else:
        if param_grid == 'auto': param_grid = {'n_clusters':clusters_range}    
        params, score, y_pred = [], [], []
        for i in param_grid['n_clusters']:
            params.append(i)
            kmeans = KMeans(n_clusters=i, n_jobs=-1, random_state=2018)
            y_pred_kms = kmeans.fit_predict(unlabeled)
            y_pred.append(y_pred_kms)
            ch_score = calinski_harabaz_score(unlabeled, y_pred_kms)
            score.append(ch_score)
        best_param = params[np.argmax(score)]
        y_pred = y_pred[np.argmax(score)]
        return get_centers(y_pred)