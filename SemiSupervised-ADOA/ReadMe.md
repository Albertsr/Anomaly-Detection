## ADOA ：Anomaly Detection with Partially Observed Anomalies

## 1. 论文地址与代码实现
- 论文地址 : [Anomaly Detection with Partially Observed Anomalies](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-ADOA/Anomaly%20Detection%20with%20Partially%20Observed%20Anomalies.pdf)

- Python实现 : [ADOA](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-ADOA/ADOA.py)；
  
- 计算聚类中心的子模块: [cluster_centers](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-ADOA/cluster_centers.py)

---

## 2. ADOA的适用场景
在只有极少量的已知异常样本(Partially Observed Anomalies)和大量的无标记数据(Unable Observations)的情况下，来进行的异常检测问题

---

## 3. 无监督方法、监督方法与PU Learning的弊端

- **若简单的形式化为无监督学习**：丢弃已有的部分标记信息会带来信息的极大损失，且效果不理想；

- **若将无标记的数据完全当作正常样本**：采用监督学习的模型来处理，则会因为引入的大量噪音导致效果欠佳；

- **PU Learning**：适用于异常值基本相似的场景，而异常样本往往千差万别，因此PU Learning的应用受到限制。

---

## 4. ADOA的处理过程

#### 4.1 阶段一：对已知异常样本聚类，并从无标签样本中过滤出潜在异常样本(Potential anomalies)**以及**可靠正常样本(Reliable Normals)

- 对于已知的异常样本进行聚类，聚类后的每一簇之间具有较高的相似性

- 对于异常样本而言，一方面，它有着容易被隔离（Isolation）的特点，另一方面，它往往与某些已知的异常样本有着较高的相似性
 
- 计算无标记样本的**隔离得分(Isolation Score)** 
  ![Isolation Score](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-ADOA/Pics/Isolation%20Score.jpg)


- 计算无标记样本与异常样本簇的**相似得分(Similarity Score)**

  ![Similarity Score](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-ADOA/Pics/Similarity%20Score.jpg)

- 计算一个样本的**异常程度总得分**

  ![TotalScore](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-ADOA/Pics/Total%20Score.jpg)

---

#### 4.2 阶段二：构建带权重的多分类模型

- 令所有已知的异常样本的权重为1
- 对于潜在异常样本，其TS(x)越高，则其作为异常样本的置信度越高，权重越大

  ![anomaly weight](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-ADOA/Pics/%E5%BC%82%E5%B8%B8%E6%9D%83%E9%87%8D.jpg)
  
- 对于可靠正常样本，其TS(x)越低，则其作为正常样本的置信度越高，权重越大
  ![nomarl weight](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-ADOA/Pics/%E6%AD%A3%E5%B8%B8%E6%9D%83%E9%87%8D.jpg)
   
---

## 5. 构建分类模型

#### 5.1 目标函数与结构风险最小化
![结构风险](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-ADOA/Pics/%E7%BB%93%E6%9E%84%E9%A3%8E%E9%99%A9.jpg)

- 对于未来的待预测样本，通过该模型预测其所属类别，若样本被分类到任何异常类，则将其视为异常样本，否则，视为正常样本；

---
