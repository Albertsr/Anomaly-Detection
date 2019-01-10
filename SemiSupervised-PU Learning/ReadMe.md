## 第一部分：PU Learning概述
### 1. PU Learning的定义
- 定义：
- 出处

---

### 2. PU Learning的三大处理方法
- 方法一：Two step strategy
  - 核心思想
  - 经典论文
  
- 方法二：Class Prior Incorporation
  - 核心思想
  - 经典论文
  
- 方法三：Biased Learning
  - 核心思想
  - 经典论文
  
---

## 第二部分：PU Learning处理方法详述

### 1. 方法一：Two step strategy

#### 1.1 核心思想
- **Step 1：** 从无标签数据集U中选出可靠的负样本集RN
- **Step 2：** 将P、RN作为训练集，训练一个分类器

#### 1.2 第一阶段：Step1

- **目的：** 筛选可靠负样本集RN

- **常用的算法：** 
  - Spy technique
  - The 1-DNF Technique
    - 通过对比P和U，从P中抽取一些在正类样本中高频出现的特征
    - U中没有或只有极少量高频特征的样本可视为可靠负样本
  - Rocchio：主要适用于文本分类算法
  - NB Classifer：If `$P(1|x)<P(Unlabelled |x)$`, then we extract example x as a reliable negative example

- **Spy technique详述**
   
   ![Spy technique](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Pics/Spy%20technique.jpg)
  
#### 1.3 第二阶段：Step2
- **目的：** 运用上一步筛选出的RN与P，训练一个分类器
- **算法：**
  - Run SVM only once on sets P and RU.
  - Run EM algorithm on P and RU.
  - Run SVM on P and RU iteratively, until no more reliable negative data can be found.
  - Run SVM on sets P and RU, and select a best
classifier in the generated models.

---

### 2. 方法二：Biased Learning
#### 2.1 核心思想
- 将无标签样本集U视为带有噪音(即正样本)的负样本集 
- 运用了代价敏感学习(cost-sensitive learning)的思想

#### 2.2 常用算法：Biased SVM
- **论文出处：** (liu et.al. 2003 )

- **核心思想**
  - 将U集视为负样本集，并赋予负样本更低的正则化参数，使得一定量的"负样本"允许被误分
  - 这些“被误分的负样本”实际上是混杂在U集中的正样本(noise)， 更低的正则化参数促进模型将这些noise被准确分类为正样本

- **BiasedSVM实际运用了代价敏感学习的思想：**   
  - 对FP赋予更低的代价，对FN赋予更高的代价
  - 负样本集由U集构成，其中有部分正样本被误标记为负。更低的FP允许模型将这些正样本“误分”为正样本，即它们真正的类别


- **最优化问题**

    ![BiasedSVM](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-PU%20Learning/Pics/BiasedSVM.jpg)  
    - 其中，C+、C− 分别为正样本与无标签样本的正则化参数，且C−小于C+
    
- **可运用权重法间接实现BiasedSVM：**      
  - 借助权重法(Weighting)，对正样本、无标签样本分别赋予更高、更低的样本权重，间接实现高低正则化参数的效果
  - fit()
  
---

### 3. 方法三：Class Prior Incorporation
#### 3.1 核心思想
- 将P集视为正样本集，**并对正样本赋予权重w(+)，且w(+)等于负样本的先验概率，即U集在整个数据集中的占比**；将U集视为负样本集，**并对负样本赋予权重w(-)，且w(-)等于P集在整个数据集中的占比**

- 对正负样本赋予权重后，可证明真实正样本被模型分类为正样本的后验概率大于0.5；真实负样本被模型分类为正样本的后验概率小于0.5；从而可以构建weighted LR模型拟合加权样本

#### 3.2 Weighted Logistic Regression
- 参数定义
  - 正样本未被标注的概率为`$\alpha$`，被标注为正样本的概率为`$1-\alpha$`
  - 将未标签样本全部视为负样本，与已知正样本的标签融合成含有噪音的类别y'(U集中的正样本被误分为负样本，因此是“含有噪音的”)
  - y为样本的真实类别，f(x)为模型对样本x的预测类别

- 期望误差之和与实际误分之和成正比例关系
```math
P(f(x)=1|y'=-1) + P(f(x)=-1|y'=1) = C(f)

P(f(x)=1|y=-1) + P(f(x)=-1|y=1) = C'(f)

```
- 最小化C(f)等价于最小化加权分类误差
```math
C(f) = P(f(x)=1|y'=-1) + P(f(x)=-1|y'=1)

= \frac{P(f(x)=1,y'=-1)}{P(y'=-1)} + \frac{P(f(x)=-1,y'=1)}{P(y'=1)}

= \frac{P(y'=1)P(f(x)=1,y'=-1) + P(y'=-1)P(f(x)=-1,y'=1)}{P(y'=-1) \cdot P(y'=1)}

\text{最小化C(f)等价于最小化加权误分数}

```

- **随机抽取的样本被分类为正、负样本的条件概率**
   - 训练集中正样本的真实概率`$\gamma = P(y=1)$`
   - 随机样本被标注为正样本的概率，即为|P|/(|P|+|U|)
   ```math
   \psi = P(y'=1) = \gamma \cdot (1-\alpha)
   ```
   - 随机样本被标注为负样本的概率
   ```math
   1-\psi = P(y'=-1) = \gamma\alpha + (1-\gamma)
   ```

- 对已标记的正样本赋予权重`$1-\psi $`，对含有噪声的负样本赋予权重`$\psi$`
  ```math
  (1-\alpha)(1-\psi) = (1-\alpha)[\gamma\alpha + (1-\gamma)]
  
  \alpha \psi = \alpha [\gamma (1-\alpha)]
   ```
- 真实正样本取正的条件概率进行标准化
  ```math
  \frac{(1-\alpha)(1-\psi)}{(1-\alpha)(1-\psi)+\alpha \psi}
  = \frac{\gamma \alpha +(1-\gamma)}{2\gamma \alpha +(1-\gamma)}
  
  \text{当$\gamma<1, \alpha<1$时，上式恒大于0.5}
  ```

- **优化函数**
   ```math
      \min \sum_{y_i=1} \frac{n(-)}{n(+)} l(y_i,g(x_i)) + \sum_{y_i=-1} l(y_i,g(x_i)) + C (\sum_{j=1}^{k} w_j^2 + b^2)
   ```
  - 其中对正样本的权重做了归一化处理
  - C为正则化参数
  
---

