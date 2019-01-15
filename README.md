# Anomaly-Detection

## 第一部分：无监督异常检测
### 1. 算法
#### 1.1 算法一：Isolation Forest
- 算法论文
- 算法解读
- 算法实现

#### 1.2 算法二：基于PCA的重构误差
- 算法论文
- 算法解读
- 算法实现
- Isolation Forest
- Local Outlier Factor 
- 基于KernelPCA的重构误差
- 基于LinearPCA的重构误差
- RobustPCC
- Mahalabonas Distance

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
- **Jupyter格式：** [detection_contrast](https://github.com/Albertsr/Anomaly-Detection/blob/master/Algo%20Contrast/%E6%97%A0%E7%9B%91%E7%9D%A3%E5%BC%82%E5%B8%B8%E6%A3%80%E6%B5%8B%E7%AE%97%E6%B3%95-%E5%AF%B9%E6%AF%94.ipynb) (建议以Jupyter运行，以更直观地显示验证过程，并对众数予以高亮显示)

#### 2.3 对比结论
- **10个随机数据集的返回结果如下图所示**

![contra_pcc](https://github.com/Albertsr/Anomaly-Detection/blob/master/Algo%20Contrast/U_contra_pcc.jpg)

- **对比结论**

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
   
