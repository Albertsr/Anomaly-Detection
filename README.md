# Anomaly-Detection
- **Author：** 马肖
- **E-Mail：** maxiaoscut@aliyun.com
- **GitHub：**  https://github.com/Albertsr

---

# 第一部分：无监督异常检测
## 1. 算法
### 1.1 孤立森林(Isolation Forest)
- **算法论文：** [Isolation Forest论文](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Isolation%20Forest/Isolation%20Forest.pdf)
- **算法解读：** [Isolation Forest个人解读](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Isolation%20Forest/ReadMe.md)
- **算法应用：** [isolationforest.py](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Isolation%20Forest/IsolationForest.py)

### 1.2 基于PCA的异常检测
- **思路一：基于样本的重构误差**  
  - **算法解读：** [Anomaly Detection Based On PCA : Chapter 1](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/ReadMe.md#chapter-1-思路一基于样本的重构误差) 
  - **算法论文：** [AI^2：Training a big data machine to defend](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Papers/AI2%20_%20Training%20a%20big%20data%20machine%20to%20defend.pdf)
  
  - **算法实现** 
    - 基于KernelPCA重构误差的异常检测： [Recon_Error_KPCA.py](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Recon_Error_KPCA.py)
    - 基于LinearPCA重构误差的异常检测： [Recon_Error_PCA.py](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Recon_Error_PCA.py)
    - 纯Numpy版本-基于LinearPCA重构误差的异常检测： [Recon_Error_PCA_Numpy_SVD.py](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Recon_Error_PCA_Numpy_SVD.py) 
      - 备注：不调用scikit-learn，只调用numpy，通过SVD实现PCA，返回结果与Recon_Error_PCA完全一致

- **思路二：基于样本在major/minor主成分上的偏离度**  
  - **算法解读：** [Anomaly Detection Based On PCA : Chapter 2](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/ReadMe.md#chapter-2-思路二基于样本在majorminor主成分上的偏离程度) 
  - **算法论文：** [A Novel Anomaly Detection Scheme Based on Principal Component ](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Papers/A%20Novel%20Anomaly%20Detection%20Scheme%20Based%20on%20Principal%20Component%20Classifier.pdf)
  - **算法实现：** [Robust_PCC.py](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Robust_PCC.py) 

- **实证分析：异常样本在最前与最后的主成分上具有最大的方差**
  - **备注：** 最前的主成分是指最大特征值对应的主成分、最后的主成分是指最小特征值对应的主成分
  - **理论分析：** [Anomaly Detection Based On PCA : Chapter 3](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/ReadMe.md#chapter-3-实证分析异常样本在最前与最后的少数几个主成分上具有最大的方差)
  - **验证代码：** [indices_ev_decrease](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/indices_ev_decrease.py)
  - **验证细节与结果：** [多个随机数据集证明异常样本在最前与最后的主成分上具有最大的方差](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/ReadMe.md#33-验证代码与结果)

### 1.3 局部异常因子(Local Outlier Factor) 
- **算法论文：** [LOF：Identifying Density-Based Local Outliers](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Local%20Outlier%20Factor/LOF%EF%BC%9AIdentifying%20Density-Based%20Local%20Outliers.pdf)
- **算法解读：** [Local Outlier Factor算法解读](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Local%20Outlier%20Factor/ReadMe.md)
- **算法应用：** [LocalOutlierFactor.py](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Local%20Outlier%20Factor/LocalOutlierFactor.py)

### 1.4 马氏距离(Mahalabonas Distance)
- **算法解读：** [Mahalanobis_Distance 算法解读](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Mahalanobis%20Distance/ReadMe.md)
- **算法实现：** 
  - **马氏距离的定义：** [mahal_dist.py](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Mahalanobis%20Distance/mahal_dist.py)
  - **马氏距离的变体：** [mahal_dist_variant.py](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Mahalanobis%20Distance/mahal_dist_variant.py)
- **马氏距离及其变体对样本异常程度的判定是一致的** 
  - **验证代码：** [verify_mahal_equivalence.py](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Mahalanobis%20Distance/verify_mahal_equivalence.py)
  - **验证方案与实验细节：**[马氏距离及其变体对样本的异常程度评估完全一致](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Mahalanobis%20Distance/ReadMe.md#4-马氏距离及其变体对样本的异常程度评估完全一致)
   
---

## 2. 性能对比
### 2.1 对比方案
- **步骤一：** 生成一系列随机数据集，每个数据集的行数(row)、列数(col)、污染率(contamination)均从某区间内随机抽取
- **步骤二：** 各个无监督异常检测算法根据指定的contamination返回异常样本的索引(anomalies_indices)
- **步骤三：** 确定baseline
  - 如果数据集中异常样本的索引已知(记为observed_anomaly_indices)，则以此作为baseline
  - 如果数据集中异常样本的索引未知，则以Isolation Forest返回的异常样本索引作为baseline
- **步骤四：** 比较各算法返回的异常样本索引与baseline的共同索引个数，个数越多，则认为此算法的检测效果相对越好
- **步骤五：** 不同的数据集对异常检测算法的性能可能会有不同的评估，因此可取众数(mode)来判定各算法的性能排序

### 2.2 对比代码 
- **Python代码：** [unsupervised_detection_contrast](https://github.com/Albertsr/Anomaly-Detection/blob/master/Algo%20Contrast/unsupervised_detection_contrast.py) (Jupyter交互式运行代码能更直观地展示验证过程)

### 2.3 对比结果
- **根据算法在特定数据集上的异常检测性能降序排列，10个随机数据集的对比结果如下图所示：**

![unsupervised_contrast](https://github.com/Albertsr/Anomaly-Detection/blob/master/Algo%20Contrast/Pics/unsupervised_contrast.jpg)

### 2.4 对比分析
#### 1）RobustPCC
- RobustPCC重点考察了样本在major/minor Principal Component上的偏差，论文作者认为异常样本在主成分空间内的投影主要集中在上述两类主成分上
- RobustPCC在构建过程中，需要通过马氏距离(的等价算法)检测并剔除数据集中一定比例(gamma)的潜在异常样本，以保证RobustPCC的有效性
- RobustPCC需要根据指定的分位点参数(quantile)来设定样本异常与否的阈值，**个人在实验中适度增大了gamma、quantile的取值，进一步降低FPR，提升鲁棒性**
- 实验结果表明，RobustPCC具有优良的异常检测性能

#### 2）Recon_Error_KPCA的检测性能显著优于Recon_Error_PCA
- 引入核函数(对比实验取RBF核)，无需显式定义映射函数，通过Kernel Trick计算样本在高维特征空间（希尔伯特空间）内的重构误差
- 高维(或无穷维)主成分空间对样本具有更强的表出能力，在低维空间内线性不可分的异常样本在高维空间内的投影将显著区别于正常样本
- 相应地，异常样本在高维(或无穷维)主成分空间内的重构误差将明显区分于正常样本，从而使得Recon_Error_KPCA的异常检测能力显著高于Recon_Error_PCA

#### 3）Isolation Forest
- Isolation Forest(孤立森林)表现稳定，在验证数据的异常索引未知情况下，个人将其预测值作为baseline，用于衡量其它算法的性能

#### 4）Mahalabonas Distance
- Mahalabonas Distance(马氏距离)实际上考虑了样本在所有主成分上的偏离度，检测性能紧跟Recon_Error_KPCA之后

#### 5）Local Outlier Factor
- LOF考虑了局部相邻密度，它存在一定的局限性：对于相关性结构较特殊的异常样本(anomalies in terms of different correlation structures)的检测能力不足

#### 备注：上述实验结论受到实验数据集的个数、样本构成等因素的影响，在实际运用中，需要根据数据集本身的特点予以灵活选择异常检测算法

---

# 第二部分：半监督异常检测
## 1. 算法
### 1.1 算法一：ADOA
- **算法论文：** [Anomaly Detection with Partially Observed Anomalies](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-ADOA/Anomaly%20Detection%20with%20Partially%20Observed%20Anomalies.pdf)
- **算法解读：** [ADOA算法解读](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-ADOA/ReadMe.md)
- **算法实现：** [ADOA](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-ADOA/ADOA.py) 【其中包含：用于返回聚类中心子模块 [cluster_centers](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-ADOA/cluster_centers.py)】

### 1.2 算法二：PU Learning
- **算法论文：** [POSTER_ A PU Learning based System for Potential Malicious URL Detection](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Papers/POSTER_%20A%20PU%20Learning%20based%20System%20for%20Potential%20Malicious%20URL%20Detection.pdf)
- **算法解读：** [PU Learning解读](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/ReadMe.md)
- **算法实现：** [pu_learning](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/pu_learning.py)

## 2. 性能对比
### 2.1 验证思路
- **思路**
  - U集中的正常样本服从正态分布
  - 已观测到的P集，以及U中混杂的异常样本由指数分布、伽马分布、卡方分布组合而成
  
### 2.2 验证代码
- **Python代码：** [SemiSupervised_detection_contrast](https://github.com/Albertsr/Anomaly-Detection/blob/master/Algo%20Contrast/semi_contrast.py)

### 2.3 对比结果
- 结果
  
  ![semi_contrast](https://github.com/Albertsr/Anomaly-Detection/blob/master/Algo%20Contrast/Pics/semi_contrast.jpg)
  
