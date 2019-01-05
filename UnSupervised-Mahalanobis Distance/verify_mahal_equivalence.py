# Author：马肖
# E-mail：maxiaoscut@aliyun.com
# Github：https://github.com/Albertsr

import numpy as np
from mahal_dist import mahal_dist
from mahal_dist_variant import mahal_dist_variant


def verify_maldist_equivalence(seed, row=1000, col=50):
    rdg = np.random.RandomState(seed)
    matrix = rdg.randn(row, col)
    
    # 马氏距离的初始定义
    dist_original = mahal_dist(matrix)
    # 根据数值大小，对数据集索引降序排列
    indices_desc_original = np.argsort(-dist_original)
    
    # 马氏距离的变体
    dist_variant = mahal_dist_variant(matrix)
    # 根据数值大小，对数据集索引降序排列
    indices_desc_variant = np.argsort(-dist_variant)
    
    assert not np.allclose(dist_original, dist_variant), '马氏距离及其变体返回的数值一般不相等'
    indices_verify_result = np.allclose(indices_desc_original, indices_desc_variant)
    return indices_verify_result

# 生成一系列不重复的随机种子
seeds = np.random.choice(range(1000), size=10, replace=False)
# 返回验证结果
verify_result = list(map(verify_maldist_equivalence, seeds))
# 输出验证结果
if all(verify_result):
    description = '经过{:}个不重复的随机数据集的测试，马氏距离及其变体对样本相对异常程度的评估是一致的'
    print(description.format(len(seeds)))
else:
    print('经过随机数据集的测试，马氏距离及其变体对样本相对异常程度的评估不一致')
