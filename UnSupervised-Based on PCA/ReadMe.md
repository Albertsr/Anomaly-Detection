- **Author：** 马肖
- **E-Mail：** maxiaoscut@aliyun.com
- **GitHub：**  https://github.com/Albertsr
- **有道云笔记版本：** [基于PCA的异常检测](http://note.youdao.com/noteshare?id=6c103b5af77b8c0c9b70d216bab60b11&sub=F02EFA86A9DC47E38A9ACDEA2C5CBB83)

---

## Chapter 1. 思路一：基于样本的重构误差

#### 1.1 论文与代码实现

- **论文地址：** [AI^2：Training a big data machine to defend](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Papers/AI2%20_%20Training%20a%20big%20data%20machine%20to%20defend.pdf)

- **基于KernelPCA重构误差的python实现：** [Recon_Error_KPCA](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Recon_Error_KPCA.py)
 
- **基于LinearPCA重构误差的python实现：** [Recon_Error_PCA](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Recon_Error_PCA.py)
  
- **纯Numpy版本：** [Recon_Error_PCA_Numpy_SVD](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Recon_Error_PCA_Numpy_SVD.py) 
    - 不调用scikit-learn，只调用numpy，通过SVD实现PCA，再进行异常检测
    - 返回结果与Recon_Error_PCA完全一致

#### 1.2 思路解析
- **靠前的主成分主要解释了大部分正常样本的方差，而靠后的主成分主要解释了异常样本的方差** 
  - 靠前的主成分是指对应于更大特征值的特征向量，靠后的主成分是指对应于更小特征值的特征向量
  - 上述特征值、特征向量可根据协方差矩阵的特征分解求得
  
  ![last_pp](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Pics/last_pp.jpg)
  
- **异常样本在靠前主成分上的投影较小，在靠后主成分上投影较大；只依靠排在前面的主成分是无法完整地重构异常样本的** 
  - 只有少量排在前面的主成分被用于矩阵重构时，异常样本引起的重构误差是要远高于正常样本的
  - 重构误差越高的样本越有可能是异常样本
  
  ![outliers_high_error](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Pics/outliers_high_error.jpg)
   
- **样本在靠后主成分上的偏差应赋予更高的权重** 
  - 令k为重构矩阵所用到的主成分数量，则随着k的逐步增加，更多靠后的主成分被用于矩阵重构
  - 这些靠后的主成分对异常样本具有更高的线性表出能力，因此样本在这些靠后的主成分上的偏差应赋予更高的权重

#### 1.3 重构矩阵的生成方式
- **重构矩阵**

 ![recon_matrix](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Pics/recon_matrix.jpg)
  
- **参数含义**  
  - R为m * n型重构矩阵，与原样本矩阵X的shape保持一致
  - k为重构矩阵过程中用到的主成分个数
  - Q为投影矩阵，其k个列向量为前k个主成分（按特征值降序排列）

#### 1.4 重构误差与异常分数
- **异常得分**  
  
  ![outlier_score](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Pics/outlierscore.jpg)
  - k表示重构矩阵所用到的主成分数，n表示主成分总数
  - ev(k)表示前k个主成分多大程度上解释了总体方差，与k值成正比

- **越靠后的主成分其对应的重构误差的权重也越大** 
  - 重构矩阵所用到的主成分越多(k值越大)，样本在靠后的主成分上的误差对应的权重ev(k)也越大
  - 靠后主成分对异常样本具有更强的表达能力，从而对应的误差应赋予更高的权重

---

## Chapter 2. 思路二：基于样本在major/minor主成分上的偏离程度
#### 2.1 论文与代码实现
- **论文地址：** [A Novel Anomaly Detection Scheme Based on Principal Component Classifier](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Papers/A%20Novel%20Anomaly%20Detection%20Scheme%20Based%20on%20Principal%20Component%20Classifier.pdf)

- **Python实现：** [Robust_PCC](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Robust_PCC.py) 

#### 2.2 术语定义
- **major principal components**
  - 将特征值降序排列后，**累计特征值之和约占50%** 的前几个特征值对应的特征向量
  - **在major principal components 上偏差较大的样本，对应于在原始特征上取极值的异常样本**
  - the observations that are outliers with respect to major principal components usually correspond to outliers on one or more
of the original variables. 

- **minor principal components**
  - 指**特征值小于0.2**对应的特征向量
  - 在minor principal components上偏差较大的样本，对应于那些**与正常样本相关性结构（the correlation structure）不一致的异常样本**
  - minor principal components are sensitive to **the observations that are inconsistent with the correlation structure of the data but
are not outliers with respect to the original variables**

- **样本在单个主成分上的偏差**
  - 样本在此特征向量上的**偏离程度定义为样本在此特征向量上投影的平方与特征值之商**
  - 其中除以特征值是为了起到归一化的作用，使得样本在不同特征向量上的偏差具有可比性
     
- **样本在所有方向上的偏差之和等价于它与样本中心之间的马氏距离**
   
   ![mahal_dist_variant](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Mahalanobis%20Distance/Pics/%E9%A9%AC%E6%B0%8F%E8%B7%9D%E7%A6%BB%E5%8F%98%E4%BD%93.jpg)

#### 2.3 算法流程
- **第一步：** 通过马氏距离筛选一定比例的极值样本从训练集中剔除，以获得鲁棒性更高的主成分及对应的特征值
  - **First, we use the Mahalanobis metric to identify the 100*gamma% extreme observations that are to be trimmed**
  - 令剩余样本构成的矩阵为remain_matrix 
  
- **第二步：** 对remain_matrix进行主成分分析，得到principal components及对应的特征值
- **第三步：** 根据特征值的取值以及相关定义，确定major principal components与minor principal components
- **第四步：** 求remain_matrix中所有样本在major principal components与minor principal components上的偏离度
- **第五步：** 根据上一步求出的两个偏离度，以及指定的分位点，求出判定样本是否异常的阈值c1与c2
- **第六步：** 对于一个待检测样本，计算它在major principal components与minor principal components上的偏离度，若其中之一超出相应的阈值则判定为异常，否则为正常样本
   
    ![classify_outlier](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Pics/classify_outlier.jpg)

#### 2.4 进一步提升RobustPCC性能的方法
- 在样本数较多的情况下，可适当提高gamma，进一步剔除训练集中较异常的样本，以提升PCC的鲁棒性
- 适当提高quantile的取值，以提升将样本判定为异常的阈值，有助于降低RobustPCC的FPR

---

## Chapter 3. 实证分析：异常样本在最前、最后的若干主成分上具有最大的方差

#### 3.1 理论分析
- 异常样本在最大以及最小的几个特征值对应的主成分上应具有更大的投影
- 若最大以及最小的几个特征值对应的主成分构成的坐标轴不存在，则**异常样本在主成分空间（the principal components’ space）无法被完整地线性表出**
- [A Novel Anomaly Detection Scheme Based on Principal Component Classifier](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Papers/A%20Novel%20Anomaly%20Detection%20Scheme%20Based%20on%20Principal%20Component%20Classifier.pdf) [Shyu, et.al, 2003]明确提出：
  - 在major principal components上偏差较大的样本，对应于在原始特征上取极值的异常样本
  - 在minor principal components上偏差较大的样本，对应于那些与正常样本相关性结构不一致的异常样本
  - 论文截图
  
   ![major_minor](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Pics/major_minor.jpg)
    
#### 3.2 验证方法
- 对数据集进行PCA，各主成分对应的特征值构成的向量记为variance_original 
- 从原数据集中剔除孤立森林（或其他异常检测算法）检测出的若干异常样本，再进行PCA，对应的特征值向量记为variance_revised 
- 计算因剔除异常样本导致的特征值变动比例delta_ratio，其中delta_ratio = (variance_revised - variance_original) / variance_original
- 找出降幅比例最大的前k（例如k=3）个特征值对应的索引indices_top_k
- 若indices_top_k中包含最小或最大的索引，则可以认为异常样本在最前与最后的少数几个主成分上具有最大的方差
 
#### 3.3 验证代码与结果
- **验证代码：** [max_ev_decrease.py](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/max_ev_decrease.py)
- **验证结果：** 经过随机生成的10个数据集的实验结果表明上述结论是正确的
- **实验细节**
  - 每个数据集均为5000 * 20型矩阵，即包含5000个样本，20个特征
  - 正常样本均服从标准正态分布，异常样本由标准伽玛分布、指数分布组合构成
  - 验证结果由下图所示，可见**降幅最大的特征值索引中至少包含最小的索引值0或最大的索引值19**
  
  ![indices_max_decrease](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Pics/indices_max_decrease.jpg)
