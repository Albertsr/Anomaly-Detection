# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
import pandas as pd
from mahal_dist import mahal_dist
from mahal_dist_variant import mahal_dist_variant

def generate_dataset(seed):
    rdg = np.random.RandomState(seed)
    row = rdg.randint(8000, 10000)
    col = rdg.randint(30, 35)
    contamination = rdg.uniform(0.015, 0.025)
    
    outlier_num = int(row*contamination)
    inlier_num = row - outlier_num
    
    # 正常样本集服从标准正态分布
    inliers = rdg.randn(inlier_num, col)
    
    # 如果outlier_num为奇数，row_1=outlier_num//2，否则row_1=int(outlier_num/2)
    row_1 = outlier_num//2 if np.mod(outlier_num, 2) else int(outlier_num/2)
    row_2 = outlier_num - row_1
    
    # outliers_sub_1服从伽玛分布；outliers_sub_2服从指数分布
    outliers_sub_1 = rdg.gamma(shape=2, scale=0.5, size=(row_1 , col))
    outliers_sub_2 = rdg.exponential(1.5, size=(row_2, col))
    outliers = np.r_[outliers_sub_1, outliers_sub_2]
    
    # 将inliers与outliers在axis=0方向上予以整合，构成实验数据集
    dataset = np.r_[inliers, outliers]
    outliers_indices = range(len(dataset))[inlier_num:]
    return dataset

def verify_maldist_equivalence(dataset):
    # 马氏距离的初始定义
    dist_original = mahal_dist(dataset)
    # 根据数值大小，对数据集索引降序排列
    indices_desc_original = np.argsort(-dist_original)
    
    # 马氏距离的变体
    dist_variant = mahal_dist_variant(dataset)
    # 根据数值大小，对数据集索引降序排列
    indices_desc_variant = np.argsort(-dist_variant)
    
    assert not np.allclose(dist_original, dist_variant), '马氏距离及其变体返回的数值一般不相等'
    indices_verify_result = np.allclose(indices_desc_original, indices_desc_variant)
    return indices_verify_result

# 生成一系列随机种子及其对应的数据集
seeds = np.random.choice(range(1000), size=10, replace=False)
datasets = list(map(generate_dataset, seeds))

# 返回验证结果
verify_result = list(map(verify_maldist_equivalence, datasets))

# 输出验证结果
if all(verify_result):
    description = '经过{:}个不重复的随机数据集的测试，马氏距离及其变体对样本相对异常程度的评估是一致的\n'
    print(description.format(len(seeds)))
else:
    print('经过随机数据集的测试，马氏距离及其变体对样本相对异常程度的评估不一致')

dataset_name = ['Dataset_' + str(i) for i in range(len(seeds))]
verify_result = pd.DataFrame(verify_result, index=dataset_name, columns=['Equivalence'])
print(verify_result.T)
