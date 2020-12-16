# Author：MaXiao
# E-mail：maxiaoscut@aliyun.com

import time
import itertools
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, Birch
from sklearn.metrics import calinski_harabasz_score


def timer(func):
    def wrapper(*args, **kwargs): 
        start = time.time()
        result = func(*args,**kwargs)
        end = time.time()
        print(func.__name__+' running time：{:.2f}s'.format(end-start))
        return result
    return wrapper

@timer
def get_cluster_centers(dataset, n_clusters='auto', cluster_algo='kmeans', params_grid='auto', random_state=2018):
    if n_clusters == 'auto':
        clusters_range = range(2, 8)
    else:
        assertion = 'n_clusters should be an integer greater than or equal to 2'
        assert isinstance(n_clusters, int) and n_clusters>=2, assertion
        clusters_range = [n_clusters]
    
    # get_centers函数可根据聚类算法对无标签数据集的预测结果(cluster_pred)返回各聚类簇的中心
    def get_centers(cluster_label, data=dataset):
        centers = []
        for label in np.unique(cluster_label):
            subset = data[cluster_label==label]
            center = np.mean(subset, axis=0)
            centers.append(center)
        return np.sort(centers, axis=0)
    
    # 通过生成参数的笛卡尔积，寻求谱聚类算法的最优参数
    if cluster_algo == 'spectral':
        if params_grid == 'auto': 
            params_grid = {'n_clusters':clusters_range, 'gamma':np.linspace(0.5, 1.5, 3)}
        params, score, y_pred_set = [], [], []
        for i, j in itertools.product(params_grid['n_clusters'], params_grid['gamma']):
            params.append((i, j))
            spectral = SpectralClustering(n_clusters=i, gamma=j, n_jobs=-1, random_state=random_state)
            y_pred_spectral = spectral.fit_predict(dataset)
            y_pred_set.append(y_pred_spectral)
            ch_score = calinski_harabasz_score(dataset, y_pred_spectral)
            score.append(ch_score)
        # 获取calinski_harabasz_score取最大值时对应的参数与预测聚类类标
        best_param = params[np.argmax(score)]
        y_pred = y_pred_set[np.argmax(score)]            
        return get_centers(y_pred), np.max(score)
    
    # 通过生成参数的笛卡尔积，寻求Birch聚类算法的最优参数
    elif cluster_algo == 'birch':
        if params_grid == 'auto':
            params_grid = {'n_clusters':clusters_range, 'branching_factor':range(2, 10), 
                           'threshold':np.linspace(0, 0.8, num=10)}
            
        params, score, y_pred_set = [], [], []
        for i, j, k in itertools.product(params_grid['n_clusters'], params_grid['branching_factor'], params_grid['threshold']):
            params.append((i, j, k))
            birch = Birch(n_clusters=i, branching_factor=j, threshold=k)
            y_pred_birch = birch.fit_predict(dataset)
            y_pred_set.append(y_pred_birch)
            ch_score = calinski_harabasz_score(dataset, y_pred_birch)
            score.append(ch_score)
        best_param = params[np.argmax(score)]
        y_pred = y_pred_set[np.argmax(score)]
        return get_centers(y_pred), np.max(score)
    
    
    # 通过生成参数的笛卡尔积，寻求DBSCAN聚类算法的最优参数
    elif cluster_algo == 'dbscan':
        if params_grid == 'auto':
            params_grid = {'eps':np.linspace(0.1, 10, num=50), 'min_samples':range(1, 10)}
            
        params, unlabeled_set, y_pred_set, score = [], [], [], []
        for i, j in itertools.product(params_grid['eps'], params_grid['min_samples']):
            dbscan = DBSCAN(eps=i, min_samples=j, n_jobs=-1)
            y_pred_dbscan = dbscan.fit_predict(dataset)
            
            # DBSCAN视预测结果为-1的样本为噪声，因此需要将“噪音样本”予以排除
            y_pred_new = y_pred_dbscan[y_pred_dbscan != -1]
            dataset_new = dataset[y_pred_dbscan != -1]
            
            # 计算剔除“噪音样本”后无标签样本的剩余比例以及聚类簇的数目
            ratio = dataset_new.shape[0] / dataset.shape[0]
            n_clusters = len(np.unique(y_pred_new))
            # 剩余样本的聚类簇数以及剩余比例满足一定要求，才能对参数及预测结果予以保留
            if n_clusters in range(2, 8) and ratio>=0.8:
                params.append((i, j))
                unlabeled_set.append(dataset_new)
                y_pred_set.append(y_pred_new)
                ch_score = calinski_harabasz_score(dataset_new, y_pred_new)
                score.append(ch_score)
        if len(score) > 0:
            best_param = params[np.argmax(score)]
            unlabeled_final = unlabeled_set[np.argmax(score)]
            y_pred = y_pred_set[np.argmax(score)]
            return get_centers(y_pred, data=unlabeled_final), np.max(score)
        else:
            descript = 'It is difficult for dbscan to determine the number of clusters in this dataset. \
            Please switch to another clustering algorithm.'
            print(descript)
    
    # 寻求Kmeans聚类算法的最优参数
    else:
        if params_grid == 'auto': params_grid = {'n_clusters': clusters_range}    
        params, score, y_pred_set = [], [], []
        for i in params_grid['n_clusters']:
            params.append(i)
            kmeans = KMeans(n_clusters=i, random_state=2018)
            y_pred_kms = kmeans.fit_predict(dataset)
            y_pred_set.append(y_pred_kms)
            ch_score = calinski_harabasz_score(dataset, y_pred_kms)
            score.append(ch_score)
        best_param = params[np.argmax(score)]
        y_pred = y_pred_set[np.argmax(score)]
        return get_centers(y_pred), np.max(score)
