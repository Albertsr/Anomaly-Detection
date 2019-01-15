# Anomaly-Detection

## 第一部分：无监督异常检测
#### 1.1 算法一：Isolation Forest
- **算法论文：** [Isolation Forest 论文](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Isolation%20Forest/Isolation%20Forest.pdf)
- **算法解读：** [Isolation Forest 算法解读](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Isolation%20Forest/ReadMe.md)
- **算法应用：** [IsolationForest.py](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Isolation%20Forest/IsolationForest.py)

#### 1.2 算法二：基于PCA的异常检测
- **思路一：基于样本的重构误差**  
  - **算法解读：** [Anomaly Detection Based On PCA](https://github.com/Albertsr/Anomaly-Detection/tree/master/UnSupervised-Based%20on%20PCA)
  - **算法论文：** [AI^2：Training a big data machine to defend](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Papers/AI2%20_%20Training%20a%20big%20data%20machine%20to%20defend.pdf)
  
  - **算法实现** 
    - 基于KernelPCA重构误差的python实现： [Recon_Error_KPCA](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Recon_Error_KPCA.py)
 
    - 基于LinearPCA重构误差的python实现： [Recon_Error_PCA](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Recon_Error_PCA.py)
  
    - 纯Numpy版本： [Recon_Error_PCA_Numpy_SVD](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Recon_Error_PCA_Numpy_SVD.py) （只调用Numpy，通过SVD实现PCA，返回结果与Recon_Error_PCA完全一致）

- **思路二：基于样本在major/minor主成分上的偏离度**  
  - **算法解读：** [Anomaly Detection Based On PCA](https://github.com/Albertsr/Anomaly-Detection/tree/master/UnSupervised-Based%20on%20PCA)
  - **算法论文：** [A Novel Anomaly Detection Scheme Based on Principal Component ](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Papers/A%20Novel%20Anomaly%20Detection%20Scheme%20Based%20on%20Principal%20Component%20Classifier.pdf)

  - **算法实现：** [Robust_PCC](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Robust_PCC.py) 

#### 1.3 算法三：Local Outlier Factor 
- **算法论文：** [LOF：Identifying Density-Based Local Outliers](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Local%20Outlier%20Factor/LOF%EF%BC%9AIdentifying%20Density-Based%20Local%20Outliers.pdf)
- **算法解读：** [Local Outlier Factor算法解读](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Local%20Outlier%20Factor/ReadMe.md)
- **算法应用：** [LocalOutlierFactor.py](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Local%20Outlier%20Factor/LocalOutlierFactor.py)

#### 1.4 算法四：Mahalabonas Distance
- **算法解读：** [Mahalanobis_Distance 算法解读](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Mahalanobis%20Distance/ReadMe.md)
- **算法实现：** 
  - **马氏距离的定义：** [mahal_dist.py](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Mahalanobis%20Distance/mahal_dist.py)
  - **马氏距离的变体：** [mahal_dist_variant.py](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Mahalanobis%20Distance/mahal_dist_variant.py)
  - **两种形式对样本异常程度判定一致：** [verify_mahal_equivalence.py](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Mahalanobis%20Distance/verify_mahal_equivalence.py)

### 2. 性能对比
#### 2.1 对比方案
- **步骤一：** 生成一系列随机数据集，每个数据集的行数(row)、列数(col)、污染率(contamination)均从某区间内随机抽取
- **步骤二：** 各个无监督异常检测算法根据指定的contamination返回异常样本的索引(anomalies_indices)
- **步骤三：** 确定baseline
  - 如果数据集中异常样本的索引已知(记为observed_anomaly_indices)，则以此作为baseline
  - 如果数据集中异常样本的索引未知，则以Isolation Forest返回的异常样本索引作为baseline
- **步骤四：** 比较各算法返回的异常样本索引与baseline的共同索引个数，个数越多，则认为此算法的检测效果相对越好
- **步骤五：** 不同的数据集对异常检测算法的性能可能会有不同的评估，因此可取众数(mode)来判定各算法的性能排序

#### 2.2 对比代码 
- **Python代码：** [unsupervised_detection_contrast](https://github.com/Albertsr/Anomaly-Detection/blob/master/Algo%20Contrast/unsupervised_detection_contrast.py)
- **Jupyter格式：** [unsupervised_detection_contrast](https://github.com/Albertsr/Anomaly-Detection/blob/master/Algo%20Contrast/%E6%97%A0%E7%9B%91%E7%9D%A3%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B%E7%AE%97%E6%B3%95-%E5%AF%B9%E6%AF%94.ipynb) (建议以Jupyter运行，以更直观地显示验证过程，并对众数予以高亮显示)

#### 2.3 对比结果与分析
- **10个随机数据集的对比结果如下图所示**

![contra_pcc](https://github.com/Albertsr/Anomaly-Detection/blob/master/Algo%20Contrast/U_contra_pcc.jpg)

- **对比分析**

- 一般来说，基于KernelPCA的重构误差**优于**基于LinearPCA的重构误差

---

## 第二部分：半监督异常检测
### 1. 算法
#### 1.1 算法一：ADOA
- 算法论文
- 算法解读
- 算法实现

#### 1.2 算法一：PU Learning
- 算法论文
- 算法解读
- 算法实现


### 2. 算法性能对比
#### 2.1 验证思路与代码
- **思路**
  - U集中的正常样本服从正态分布
  - 已观测到的P集，以及U中混杂的异常样本由指数分布、伽马分布、卡方分布组合而成
  
- **代码**
  - [semi_contrast](https://github.com/Albertsr/Anomaly-Detection/blob/master/Algo%20Contrast/semi_contrast.py)

#### 2.2 对比结果
- 结果
  
  ![semi_contra](https://github.com/Albertsr/Anomaly-Detection/blob/master/Algo%20Contrast/semi_contra.jpg)
   
