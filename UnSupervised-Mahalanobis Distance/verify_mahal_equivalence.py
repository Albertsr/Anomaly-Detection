# Author：MaXiao
# E-mail：maxiaoscut@aliyun.com

import numpy as np
import pandas as pd
from mahal_dist import cal_mahal_dist
from mahal_dist_variant import cal_mahal_dist_variant

def generate_dataset(seed):
    rdg = np.random.RandomState(seed)
    row = rdg.randint(8000, 10000)
    col = rdg.randint(30, 35)
    contamination = rdg.uniform(0.015, 0.025)
    
    outlier_num = int(row*contamination)
    inlier_num = row - outlier_num
    
    # the normal sample set obeys the standard normal distribution.
    inliers = rdg.randn(inlier_num, col)
    
    # If outlier_num is odd, row_1=outlier_num//2，else row_1=int(outlier_num/2)
    row_1 = outlier_num//2 if np.mod(outlier_num, 2) else int(outlier_num/2)
    row_2 = outlier_num - row_1
    
    # outliers_sub_1 obeys gamma distribution and outliers_sub_2 obeys exponential distribution.
    outliers_sub_1 = rdg.gamma(shape=2, scale=0.5, size=(row_1 , col))
    outliers_sub_2 = rdg.exponential(1.5, size=(row_2, col))
    outliers = np.r_[outliers_sub_1, outliers_sub_2]
    
    dataset = np.r_[inliers, outliers]
    outliers_indices = range(len(dataset))[inlier_num:]
    return dataset

def verify_maldist_equivalence(dataset):
    mahal_dist = cal_mahal_dist(dataset)
    indices_desc = np.argsort(-mahal_dist)
    
    mahal_dist_variant = cal_mahal_dist_variant(dataset)
    indices_desc_variant = np.argsort(-mahal_dist_variant)
    
    square_bool = np.allclose(mahal_dist**2, mahal_dist_variant)
    indices_bool = np.all(indices_desc==indices_desc_variant)
    return square_bool and indices_bool


seeds = np.random.choice(range(1000), size=10, replace=False)
datasets = [generate_dataset(seed) for seed in seeds]
bool_results = [verify_maldist_equivalence(dataset) for dataset in datasets]

'''
relevant conclusions
- the square of Mahalanobis distance is equal to its variation
- they are consistent in determining the abnormal degree of the sample
'''

if all(bool_results):
    print('Right! The relevant conclusions about Mahalanobis distance are correct.')
else:
    print('Wrong! The relevant conclusions about Mahalanobis distance are incorrect.')

dataset_name = ['Dataset_' + str(i) for i in range(len(seeds))]
verify_result = pd.DataFrame(bool_results, index=dataset_name, columns=['Equivalence'])
print(verify_result.T)
