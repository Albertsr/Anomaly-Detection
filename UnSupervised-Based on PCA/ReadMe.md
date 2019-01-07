- **Author：** 马肖
- **E-Mail：** maxiaoscut@aliyun.com
- **GitHub：**  https://github.com/Albertsr

---
### 由于GitHub无法正常显示LaTeX公式，请点击以下链接，查看完整版本

### 完整版本： [基于PCA的异常检测](http://note.youdao.com/noteshare?id=6c103b5af77b8c0c9b70d216bab60b11&sub=F02EFA86A9DC47E38A9ACDEA2C5CBB83)

---
<br>

## 1. 基于PCA的异常检测有以下两种思路

#### 1.1 关于思路一的简述
- **核心思想：** 通过PCA将数据映射到低维特征空间，在低维特征空间不同维度上的偏差越大的样本越有可能是异常样本；

- **论文地址：** [A Novel Anomaly Detection Scheme Based on Principal Component Classifier](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Papers/A%20Novel%20Anomaly%20Detection%20Scheme%20Based%20on%20Principal%20Component%20Classifier.pdf)

- **Python实现：** [Robust_PCC](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Robust_PCC.py) 

#### 1.2 关于思路二的简述
- **核心思想：** 将数据映射到低维特征空间，然后尝试用低维特征重构原始数据，重构误差越大的样本越有可能是异常样本；

- **论文地址：** [AI^2：Training a big data machine to defend](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Papers/AI2%20_%20Training%20a%20big%20data%20machine%20to%20defend.pdf)
   
- **基于KernelPCA重构误差的python实现：** [Recon_Error_KPCA](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Recon_Error_KPCA.py)
  
- **基于LinearPCA重构误差的python实现** 
  - **推荐版本：** [Recon_Error_PCA](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Recon_Error_PCA.py)
  - **纯Numpy版本：** [Recon_Error_PCA_Numpy_Only](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Recon_Error_PCA_Numpy_Only.py) (只调用Numpy，通过SVD实现PCA，再进行异常检测)

---

## 2. 思路一：基于样本在各主成分上的偏离程度
#### 2.1 术语定义
- **major principal components**
  - 将特征值降序排列后，累计特征值之和约占50%的前几个特征值对应的特征向量
  - 在major principal components 上偏差较大的样本，对应于在原始特征上取极值的异常样本
  - the observations that are outliers with respect to major principal components usually correspond to outliers on one or more
of the original variables. 

- **minor principal components**
  - 指特征值小于0.2对应的特征向量
  - 在minor principal components上偏差较大的样本，对应于那些与正常样本相关性结构不一致的异常样本
  - minor principal components are sensitive to the observations that are inconsistent with the correlation structure of the data but
are not outliers with respect to the original variables

- **样本在单个主成分上的偏差**
  - 样本在此特征向量上的偏离程度定义为样本在此特征向量上投影的平方与特征值之商

  - 其中除以特征值是为了起到归一化的作用，使得样本在不同特征向量上的偏差具有可比性
 
- **样本在所有方向上的偏差之和等价于它与样本中心之间的马氏距离**
  - 当偏差之和大于某个阈值时，便可判定为异常样本


#### 2.2 算法流程
- **第一步：** 通过马氏距离筛选一定比例的极值样本从训练集中剔除，以获得鲁棒性更高的主成分及对应的特征值
  - **First, we use the Mahalanobis metric to identify the 100*gamma% extreme observations that are to be trimmed**
  - 设剩余样本构成的矩阵为remain_matrix 
  
- **第二步：** 对remain_matrix进行主成分分析，得到主成分及对应的特征值
- **第三步：** 根据上一步求出的特征值，确定major principal components与minor principal components
- **第四步：** 求remain_matrix中所有样本在major principal components与minor principal components上的偏离度
- **第五步：** 根据指定的分位点和上一步求出的两个偏离度向量，求出判定样本是否为异常的阈值c1与c2
- **第六步：** 对应一个待测样本，计算它在major principal components与minor principal components上的偏离度，若其中之一超出对应的阈值即为异常，否则为正常样本
   
    ![classify_outlier](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Pics/classify_outlier.jpg)

#### 2.3 进一步提升Robust_PCC性能的方法
- 在样本数较多的情况下，可适当提高gamma，以提升PCC的鲁棒性
- 适当提高quantile的取值，以提升将样本判定为异常的阈值，有助于降低Robust_PCC的FPR

---

## 3. 思路二：基于样本的重构误差

#### 3.1 思路解析
- 根据特征值的大小，对主成分降序排列，则**靠前的主成分主要解释了大部分正常样本的方差，而靠后的主成分主要解释了异常样本的方差**
   
  ![last_pp](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Pics/last_pp.jpg)
  
- **异常样本在靠前的主成分上的分量很小，仅仅只靠排在前面的主成分是无法完整地将异常样本线性表出的。** 因此，只有少量排在前面的主成分被用于矩阵重构时，异常样本引起的重构误差是要远高于正常样本的，且重构误差越高的样本越有可能是异常样本
  
  ![outliers_high_error](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Pics/outliers_high_error.jpg)
   

- 令重构初始特征空间选取的主成分数量为k，则k增加到一定程度后，正常样本重构误差的减小幅度越来越不明显，而异常样本重构误差的减小幅度会越来越明显，因为越靠后的主成分对异常样本的解释力越强。因此，**重构误差对应的权重应与k成正比，使得异常样本的加权重构误差明显高于正常样本的重构误差。**

#### 3.2 重构矩阵的生成方式
 
 ![recon_matrix](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Pics/recon_matrix.jpg)

#### 3.3 重构误差与异常分数


---

## 4. 实证分析：异常样本在最前与最后的少数几个主成分上具有最大的方差

#### 4.1 相关结论
- 异常样本在最大以及最小的几个特征值对应的主成分上应具有更大的分量
- 若最大以及最小的几个特征值对应的主成分构成的坐标轴不存在，则异常样本无法被完整地线性表出
- Mei-Ling Shyu等人也在论文[A Novel Anomaly Detection Scheme Based on Principal Component Classifier](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Papers/A%20Novel%20Anomaly%20Detection%20Scheme%20Based%20on%20Principal%20Component%20Classifier.pdf)明确提出：
  - 在major principal components上偏差较大的样本，对应于在原始特征上取极值的异常样本
  - 在minor principal components上偏差较大的样本，对应于那些与正常样本相关性结构不一致的异常样本
  - ![major_minor](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Pics/major_minor.jpg)
    
#### 4.2 验证方法
- 对数据集进行PCA，各主成分对应的特征值构成的向量记为variance_original （已降序排列）
- 从原数据集中剔除孤立森林（或其他异常检测算法）检测出的若干异常样本，再进行PCA，对应的特征值向量记为variance_revised （已降序排列）
- 计算各特征值的降幅比例delta_ratio，其中delta_ratio = (variance_revised - variance_original) / variance_original
- 找出降幅比例最大的前k（例如k=3）个特征值对应的索引indices_top_k
- 若indices_top_k中包含最小或最大的索引，则可以认为异常样本在最前与最后的少数几个主成分上具有最大的方差

#### 4.3 验证代码与结果
- **验证代码：** [variance_contrast](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/variance_contrast.py)
- 经过随机生成的10个数据集的实验结果表明上述结论是正确的，实验相关细节如下：
  - 实验数据集均为5000*20的矩阵
  - 正常样本服从标准正态分布，异常样本由泊松分布、指数分布组合构成
  - 实验数据集均为20列，由下图可见降幅最大的索引中至少包含最小的索引值0或最大的索引值19
  
  ![verify_result](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Pics/verify_result.jpg)
