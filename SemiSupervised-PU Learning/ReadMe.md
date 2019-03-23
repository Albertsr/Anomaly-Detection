## 第一部分：PU Learning概述
### 1. PU Learning的定义
-  P：Positive，表示正样本集 
-  U：Unlabeled，表示无标签样本集
-  即存在正样本集、无标签样本集，不存在负样本集情况下的分类模型问题

---

### 2. PU Learning的三大处理方法

- **方法一：** Two Step Strategy
- **方法二：** Class Prior Incorporation
- **方法三：** Biased Learning

- 论文出处：[Learning From Positive and Unlabeled Data：A Survey](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Papers/Learning%20From%20Positive%20and%20Unlabeled%20Data%EF%BC%9AA%20Survey.pdf)

![three catogary](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Pics/three%20cate.jpg)


---

## 第二部分：PU Learning处理方法详述

### 1. 方法一：Two Step Strategy

#### 1.1 核心思想
- **Step 1：** 从无标签数据集U中选出可靠的负样本集RN(Reliable Negatives)
- **Step 2：** 将P、RN作为训练集，训练一个分类器，然后对U集进行预测

#### 1.2 第一阶段(Step1)

- **目的：** 筛选可靠负样本集**RN(Reliable Negatives)**

- **常用的算法：** 
  - **Spy technique**
  - **The 1-DNF Technique：** 通过对比P和U，从P中抽取一些在正类样本中高频出现的特征，U中没有或只有极少量高频特征的样本可视为可靠负样本
  - **Rocchio：** 主要适用于文本分类算法
  - **NB Classifer：** 若P(1|x)<P(Unlabelled|x), 则样本可视为可靠负样本

- **Spy technique详述**
   - 论文地址：[A PU Learning based System for Potential Malicious URL Detection](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Papers/POSTER_%20A%20PU%20Learning%20based%20System%20for%20Potential%20Malicious%20URL%20Detection.pdf)
   - 代码实现：[pu_learning](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/pu_learning.py)
   
   ![Spy technique](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Pics/Spy%20technique.jpg)
   
  
#### 1.3 第二阶段(Step2)
- **目的：** 运用上一步筛选出的RN与P，训练一个分类器
- **算法：** 可根据数据集特点灵活选择

---

### 2. 方法二：Biased Learning
#### 2.1 核心思想
- 将无标签样本集U视为带有噪音(即正样本)的负样本集 
- 运用了代价敏感学习(cost-sensitive learning)的思想

#### 2.2 常用算法：Biased SVM
- **论文：** [Building Text Classifiers Using Positive and Unlabeled Examples](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Papers/Building%20Text%20Classifiers%20Using%20Positive%20and%20Unlabeled%20Examples.pdf)

- **算法实现：** [biased_svm.py](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/biased_svm.py)

- **核心思想**
  - 将U集视为负样本集，并赋予负样本更低的正则化参数，使得一定量的"负样本"允许被误分
  - 这些“被误分的负样本”实际上是混杂在U集中的正样本(noise)， 更低的正则化参数促进模型将这些noise被准确分类为正样本

- **BiasedSVM实际运用了代价敏感学习的思想：**   
  - 对FP赋予更低的代价，对FN赋予更高的代价
  - 负样本集由U集构成，其中有部分正样本被误标记为负。更低的FP代价允许模型将这些正样本“误分”为正样本，即它们真正的类别


- **最优化问题**

    ![BiasedSVM](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Pics/BiasedSVM.jpg)  
    - 其中，C+、C− 分别为正样本与无标签样本的正则化参数，且C−小于C+
    
- **可运用权重法间接实现BiasedSVM：**      
  - 借助权重法(Weighting)，对正样本、无标签样本分别赋予更高、更低的样本权重，间接实现高低正则化参数的效果
  
---

### 3. 方法三：Class Prior Incorporation
#### 3.1 核心思想
- 将P集视为正样本集，**并对正样本赋予权重w(+)，且w(+)等于负样本的先验概率，即U集在整个数据集中的占比**；
- 将U集视为负样本集，**并对负样本赋予权重w(-)，且w(-)等于P集在整个数据集中的占比**
- 对正负样本赋予权重后，可证明真实正样本被模型分类为正样本的后验概率大于0.5；真实负样本被模型分类为正样本的后验概率小于0.5；从而可以构建weighted LR模型拟合加权样本

#### 3.2 Weighted Logistic Regression
- **论文：** [Learning with Positive and Unlabeled Examples Using Weighted Logistic Regression](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Papers/Learning%20with%20Positive%20and%20Unlabeled%20Examples%20Using%20Weighted%20Logistic%20Regression.pdf)

- **算法实现：** [weighted_lr.py](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/weighted_lr.py)

- **参数定义**
  
  ![param](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Pics/param.jpg)

- **期望误差之和C(f)与实际误分之和C'(f)成正比例关系**

  ![relation](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Pics/relation.jpg)

- **最小化C(f)等价于最小化加权分类误差**

   ![minc](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Pics/minC.jpg)

- **随机抽取的样本被分类为正、负样本的条件概率**
   
   ![post_prob](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Pics/post_prob.jpg)


 ![optimal](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Pics/optimal.jpg)

---
### 附录：关于Spy Technique中抽样比例sample_ratio的总结

- **1）sample_ratio过高时：**
  - **模型将样本预测为正的概率`$P(1|x)$`偏小：** 模型为了将spy样本分类为负样本，决策边界会显著趋近于正样本集，表现为`$P(1|x)$`偏小；无标签样本属于正类别的概率值`$P(1|x \in U)$`的最大值、平均值的趋势曲线max_prob、avg_prob也说明了这一点

  - **`$P(1|x)$`偏小导致FN增加FP减小，进一步使得Recall下降，Specificity和Precision上升：** Recall曲线、Specificity曲线、Precision曲线的趋势证明了这一点

- **2）sample ratio较小时，生成的theta是否适宜与数据集本身的特点有关**
  - spy样本也许具有非常明显的正样本特质，模型为了将其预测为负样本，决策样本大幅向正样本移动，导致theta偏低，RN的样本数偏多，混入了较多的正样本
  - spy样本的正类特质不明显，决策边界无需大幅调整便可将其预测为负样本，生成的theta随机性较大
  
- **3）sample ratio的经验取值为0.15**
  - 论文《Partially Supervised Classification of Text Documents》中比较了5%、10%、20%, 结果相差不大

- **4）sample ratio在一定区间内变化时，模型对正负样本的区分能力可能处于震荡状态**
  - 这种情况下，临近模型决策边界两边的正负样本相似度越来越高，模型难以准确地学习到理想决策边界，表现为AUC等指标上下震荡

  ![sample_ratio_rf](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Pics/sample_ratio_rf.png)

---

### Reference
- [A Survey on Postive and Unlabelled Learning](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Papers/A%20Survey%20on%20Postive%20and%20Unlabelled%20Learning.pdf)
- [Learning From Positive and Unlabeled Data：A Survey](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Papers/Learning%20From%20Positive%20and%20Unlabeled%20Data%EF%BC%9AA%20Survey.pdf)
- [Building Text Classifiers Using Positive and Unlabeled Examples](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Papers/Building%20Text%20Classifiers%20Using%20Positive%20and%20Unlabeled%20Examples.pdf)
- [POSTER_ A PU Learning based System for Potential Malicious URL Detection](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Papers/POSTER_%20A%20PU%20Learning%20based%20System%20for%20Potential%20Malicious%20URL%20Detection.pdf)
- [Partially Supervised Classification of Text Documents](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Papers/Partially%20Supervised%20Classification%20of%20Text%20Documents.pdf)
- [Learning with Positive and Unlabeled Examples Using Weighted Logistic Regression](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Papers/Learning%20with%20Positive%20and%20Unlabeled%20Examples%20Using%20Weighted%20Logistic%20Regression.pdf)
- [Learning from Positive and Unlabeled Examples with Different Data Distributions](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Papers/Learning%20from%20Positive%20and%20Unlabeled%20Examples%20with%20Different%20Data%20Distributions_2005_A_EM.pdf)
