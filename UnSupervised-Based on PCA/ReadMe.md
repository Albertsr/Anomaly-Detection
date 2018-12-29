- **Author：** 马肖
- **E-Mail：** maxiaoscut@aliyun.com
- **GitHub：**  https://github.com/Albertsr

---

## 1. 基于PCA的异常检测有以下两种思路

#### 1.1 关于思路一的简述
- **核心思想：** 通过PCA将数据映射到低维特征空间，在低维特征空间不同维度上的偏差越大的样本越有可能是异常样本；

- **论文地址：** [A Novel Anomaly Detection Scheme Based on Principal Component Classifier](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Papers/A%20Novel%20Anomaly%20Detection%20Scheme%20Based%20on%20Principal%20Component%20Classifier.pdf)

- **python实现：** [Robust_PCC](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Robust_PCC.py) 


#### 1.2 关于思路二的简述
- **核心思想：** 将数据映射到低维特征空间，然后尝试用低维特征重构原始数据，重构误差越大的样本越有可能是异常样本；

- **论文地址：** [AI^2：Training a big data machine to defend](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Papers/AI2%20_%20Training%20a%20big%20data%20machine%20to%20defend.pdf)

- **python实现：**
  - 基于LinearPCA的重构误差：
    - 推荐版本：[Recon_Error_PCA](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Recon_Error_PCA.py)
    - 练习版本：[Recon_Error_PCA_Numpy_Only](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Recon_Error_PCA_Numpy_Only.py) (只调用Numpy实现，通过协方差矩阵的特征值分解进行PCA)

  - 基于Kernel PCA的重构误差的python实现：[Recon_Error_KPCA](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Recon_Error_KPCA.py)

---

## 2. 思路一：基于样本在各主成分上的偏离程度
#### 2.1 术语定义
- **major principal components**
  - 将特征值降序排列后，累计特征值之和约占50%的前几个特征值对应的特征向量
  - 在major principal components 上偏差较大的样本，对应于在原始特征上取极值的异常样本

- **minor principal components**
  - 指特征值小于0.2对应的特征向量
  - 在minor principal components上偏差较大的样本，对应于那些与正常样本相关性结构不一致的异常样本（observations that do not have the same correlation structure as the normal instances）

- **样本在单个主成分上的偏差**
  - 设`$e_j$`为低维空间的一个主成分（特征向量），则样本`$x_i$`在此特征方向上的偏离程度定义为 `$d_{ij} = (x_i^T e_j)^2 / \lambda_j$` 

  - 其中除以特征值`$\lambda_j$`是为了起到归一化的作用，使得样本在不同特征向量上的偏差具有可比性
 
- **样本在所有方向上的偏差之和等价于它与样本中心之间的马氏距离**
  - `$Score(x_i) = \sum_{j=1}^{n} d_{ij} = \sum_{j=1}^{n} (x_i^T e_j)^2 / \lambda_j $`
  - 当`$Score(x_i)$`大于某个阈值时，便可将样本`$x_i$`判定为异常样本
   
  - 论文原文
  
    ![马氏距离](7755427582DE4460BDB21313FF04985B)

#### 2.2 算法流程
- **第一步：** 通过马氏距离筛选一定比例的极值样本从训练集中剔除，以获得鲁棒性更高的主成分及对应的特征值，剩余样本构成的矩阵记为remain_matrix 
  - 论文原文：First, we use the Mahalanobis metric to identify the 100`$\gamma$`% extreme observations that are to be trimmed
 
- **第二步：** 求remain_matrix的协方差矩阵，以及此协方差矩阵的特征向量与特征值；即对remain_matrix进行主成分分析

- **第三步：** 根据上一步求出的特征值，确定major principal components与minor principal components，及对应的特征值

- **第四步：** 求remain_matrix中所有样本在major principal components与minor principal components上的偏离度，公式如下：
  ```math
  \sum_{i=1}^{q} \frac{y_i^2}{\lambda_i} ,\ \text{其中$q$为major principal components的数量；} 
  \sum_{i=p-r+1}^{p} \frac{y_i^2}{\lambda_i},\  \text{其中$r$为minor principal components的数量}
    
  \text{$y_i = x^T e_i$表示样本$x$在$e_i$上的投影，   $p$为主成分总数}
  ```
- **第五步：** 根据指定的分位点和上一步求出的两个偏离度向量，求出判定样本是否为异常的阈值

- **第六步：** 对应一个给定的待测样本，计算它在major principal components与minor principal components上的偏离度，若其中之一超出对应的阈值即为异常，否则为正常样本

  ![判定](F16A540553F04717B4C81AEC653D5B1E)

---

## 3. 思路二：基于样本的重构误差

#### 3.1 思路解析

- 异常样本具有**few and different**的特点，即异常样本占比较少，且特征构成与正常样本不一致。根据特征值的大小，对主成分降序排列，则靠前的主成分解释了大部分正常样本的方差，而最后的主成分主要解释了异常样本的方差。
![Inkedshouwei_LI](5430C79B4BE649E6BF0D4E30B4478536)


- 若只选取靠前的主成分用于重构初始特征空间，则异常样本无法被完整表出，异常样本引起的重构误差要远高于正常样本。因此，**在只运用部分靠前的主成分用于重构矩阵时，重构误差越高的样本越有可能是异常样本**

- 令重构初始特征空间选取的主成分数量为`$k$`，则`$k$`增加到一定程度后，正常样本重构误差的减小幅度越来越不明显，而异常样本重构误差的减小幅度会越来越明显，因为越靠后的主成分对异常样本的解释力越强。因此，**重构误差对应的权重应与k成正比，使得异常样本的加权重构误差明显高于正常样本的重构误差。**
---

#### 3.2 重构矩阵与重构样本

- **重构矩阵：** `$R^{k}_{m*n} = X_{m*n}Q_{n*k}Q^T_{k*n}$`，其中`$Q$`为投影矩阵，由协方差矩阵的特征向量构成；`$k$`为重构矩阵过程中用到的主成分个数；

- **重构样本：** 对于数据样本`$x_i$`，其重构样本`$R_{i}^{k}$`为重构矩阵`$R^{k}$`的第`$i$`行
---

#### 3.3 异常分数计算公式
- **样本`$x_i$`的异常得分:** 
  
  `$Score(x_i) = \sum_{k=1}^{n} (|x_i - R_{i}^{k}|) * ev(k)$`,其中`$ev(k) = \sum_{j=1}^{k} \lambda_j / \sum_{j=1}^{n} \lambda_j$`

- **特征值与对应主成分上的偏差权重成反比:**   
`$ev(k)$`表示前`$k$`个主成分多大程度上解释了总体方差，与`$k$`值成正比。这也意味着重构矩阵用到的主成分越多，偏差`$|x_i - R_{i}^{k}|$`的权重越大
