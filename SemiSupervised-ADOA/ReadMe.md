## ADOA ：Anomaly Detection with Partially Observed Anomalies

## 1. ADOA的适用场景
在只有极少量的已知异常样本(Partially Observed Anomalies)和大量的无标记数据(Unable Observations)的情况下，来进行的异常检测问题

---

## 2. 无监督方法、监督方法与PU Learning的弊端

- **若简单的形式化为无监督学习**：丢弃已有的部分标记信息会带来信息的极大损失，且效果不理想；

- **若将无标记的数据完全当作正常样本**：采用监督学习的模型来处理，则会因为引入的大量噪音导致效果欠佳；

- **PU Learning**：适用于异常值基本相似的场景，而异常样本往往千差万别，因此PU Learning的应用受到限制。

---

## 3.ADOA的处理过程

#### 3.1 阶段一：对已知异常样本聚类，并从无标签样本中过滤出潜在异常样本(Potential anomalies)**以及**可靠正常样本(Reliable Normals)

- 对于已知的异常样本进行聚类，聚类后的每一簇之间具有较高的相似性

- 对于异常样本而言，一方面，它有着容易被隔离（Isolation）的特点，另一方面，它往往与某些已知的异常样本有着较高的相似性
 
- 计算无标记样本的**隔离得分(Isolation Score)** 
  ![Isolation Score](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-ADOA/Pics/Isolation%20Score.jpg)


- 计算无标记样本与异常样本簇的**相似得分**（Similarity Score）
```math
SS(x) = \underbrace{max}_{i} \ e^{-(x - \mu_i)^2}
\text{，其中} \mu_i \text{为已知异常样本簇的中心}
```

- 计算一个样本的**异常程度总得分**
```math
TS(x) = \theta * IS(x) + (1-\theta) * SS(x)
\text{, 其中权重参数} \theta \in [0, 1]

\text{若$TS(x) \geq \alpha$,\ 则判定为潜在异常样本；} 
\text{若$TS(x) \leq \beta$,\  则判定为可靠正常样本}
```
---

#### 3.2 阶段二：构建带权重的多分类模型

- 令所有已知的异常样本的权重为1
- 对于潜在异常样本，其TS(x)越高，则其作为异常样本的置信度越高，权重越大
   
```math
w(x) = \frac{TS(x)}{\max TS(x)}
```
- 对于可靠正常样本，其TS(x)越低，则其作为正常样本的置信度越高，权重越大

```math
w(x) = \frac{\max TS(x) - TS(x)} {\max TS(x) - \min TS(x)}
```
---

## 4. 构建分类模型

#### 4.1 目标函数与结构风险最小化

```math
\min\ \sum_{i} w_i * l(y_i, f(x_i)) + \lambda R(w) 
\text{其中，$w_i$为样本权重，$l$为损失函数，$R(w)$为正则项；}

\text{论文采用SVM算法，即损失函数为$Hingeloss=ReLU(1-y_i \cdot \hat{y}_i$)，正则项为权重向量$w$的L2范数: $||w||^2$}
```

- 对于未来的待预测样本，通过该模型预测其所属类别，若样本被分类到任何异常类，则将其视为异常样本，否则，视为正常样本；

---
