- **Author：** 马肖

- **E-Mail：** maxiaoscut@aliyun.com

- **GitHub：**  https://github.com/Albertsr

- **请点击以下有道云笔记链接，查看详细分析与论文解读：**

  [详细分析与论文解读 : 基于PCA的异常值检测——by MaXiao](http://note.youdao.com/noteshare?id=1ed243124672faf551db23f651161b37&sub=6470C23BA9B540E2B3CAC75FD25642CA).

---

### 1. 基于PCA的异常值检测有两种思路：

- **思路1：** 
  - 通过PCA将数据映射到低维特征空间，在低维特征空间不同维度上的偏差越大的样本越有可能是异常样本；
  - 论文地址：[A Novel Anomaly Detection Scheme Based on Principal Component Classifier](https://github.com/Albertsr/Anomaly-Detection/blob/master/Anomaly%20Detection%20based%20on%20PCA/Papers/A%20Novel%20Anomaly%20Detection%20Scheme%20Based%20on%20Principal%20Component%20Classifier%EF%BC%9A2003.pdf)
  
- **思路2：** 
  - 将数据映射到低维特征空间，然后尝试用低维特征重构原始数据，重构误差越大的样本越有可能是异常样本；
  - 论文地址：[AI^2：Training a big data machine to defend](https://github.com/Albertsr/Anomaly-Detection/blob/master/Anomaly%20Detection%20based%20on%20PCA/Papers/AI2%20_%20Training%20a%20big%20data%20machine%20to%20defend.pdf)

---

### 2. 思路一：基于样本在各主成分上的偏离程度
- Python代码实现
  - [基于各主成分上累计偏差的python实现](https://github.com/Albertsr/Anomaly-Detection/blob/master/Anomaly%20Detection%20based%20on%20PCA/PCA_Mahalanobis.py)
  - [基于Major与Minor主成分的异常值检测的python实现](https://github.com/Albertsr/Anomaly-Detection/blob/master/Anomaly%20Detection%20based%20on%20PCA/PCA_Major_Minor.py)

- 思路分析
  - 样本点各个主成分上的累计偏差过高或过低，都有可能是异常样本；
  - 在靠前的主成分上偏差较大的样本，对应于在某些原始特征上取极值的异常样本；在靠后的主成分上偏差较大的样本，对应于那些与正常样本相关性结构不一致的异常样本；

---

### 3. 思路二：基于样本的重构误差
- Python代码实现
  - 基于线性PCA重构误差的python实现：
    - 版本1：[PCA_Recon_Error](https://github.com/Albertsr/Anomaly-Detection/blob/master/Anomaly%20Detection%20based%20on%20PCA/PCA_Recon_Error.py) ;
    - 版本2：[PCA_Recon_Error_Numpy_Only](https://github.com/Albertsr/Anomaly-Detection/blob/master/Anomaly%20Detection%20based%20on%20PCA/PCA_Recon_Error_Numpy_Only.py) 只调用Numpy，手动通过协方差矩阵的特征值分解实现PCA，sklearn提供的API是通过SVD实现PCA

  - 基于Kernel PCA重构误差的python实现：[KPCA_Recon_Error](https://github.com/Albertsr/Anomaly-Detection/blob/master/Anomaly%20Detection%20based%20on%20PCA/KPCA_Recon_Error.py)

- 思路分析
  - 异常样本具有**few and different**的特点，即异常样本占比较少，且特征构成与正常样本不一致。根据特征值的大小，对主成分降序排列，则靠前的主成分解释了大部分正常样本的方差，而最后的主成分主要解释了异常样本的方差。

  - 若只选取靠前的主成分用于重构初始特征空间，则异常样本无法被完整表出，异常样本引起的重构误差要远高于正常样本。因此，**重构过程中选取的主成分较少的情况下，重构误差越高的样本越有可能是异常样本**。

  - 假设重构初始特征空间选取的主成分数量为`$k$`，则`$k$`增加到一定程度后，正常样本重构误差的减小幅度越来越不明显，而异常样本重构误差的减小幅度会越来越明显。因此，**重构误差对应的权重应与k成正比，使得异常样本的加权重构误差明显高于正常样本的重构误差。**
