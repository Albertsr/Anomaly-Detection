- **Author：** 马肖
- **E-Mail：** maxiaoscut@aliyun.com
- **GitHub：** https://github.com/Albertsr

---

#### 1.iForest的核心思想
- 异常点相比其他数据点较为疏离，只需少数几次切分就可以被隔离，即异常样本更容易被划分至叶结点，从而使得异常样本所属的叶结点距离根节点有更短的路径长度。
- 在iTree中，异常点被isolated之后更加靠近树的根部，而正常数据isolated之后在树中更深

---

#### 2.概述
- iForest 适用于**连续性数据**的异常值检测，属于**非参数（无数学模型）、无监督**模型


- **iForest利用了异常样本的两个特点**：
  - **few** : 异常样本在样本集中占比较小
  - **different** : 异常样本的某些特征的取值明显区别于正常样本

- iForest只有两个参数：**iTree的个数、 训练每棵iTree的样本数**
  
  - iTree是**二叉树结构**，iTree的个数默认取100，论文原文：path lengths usually converge well before t = 100

  - **运用小样本集训练单颗itree有助于减轻swamping and masking effect**
  
    - **swamping**：是指将正常样本识别为异常样本，类似于FP；
    - **masking**：是指异常样本没有被识别出来，类似于FN；
    - swamping与masking更容易在数据量较大的情况下出现，因此训练单棵iTree的样本数不宜过多，默认不超过256
    - 大样本集不一定增强其性能，反而会增加计算量和内存占用

- 在n个训练样本均不相同的情况下，训练出的iTree具有n个叶结点，n-1个内部结点(非叶结点)，总结点数为2n-1

---

#### 3.iForest的训练过程

- 抽取若干个样本构成子样本集，放置于根节点，用于训练单颗iTree

- 随机选择一个特征q作为起始结点，然后在特征q的最大值和最小值之间随机选择一个值p作为分割点

- 根据属性q的取值进行分枝，把q<p 的样本划分至左子节点，把 q>=p的样本划分至右子节点

- 重复上两步，递归地构造左子节点和右子节点，直到满足以下条件之一：
  - 数据不可再分，即：只包含单个样本，或全部样本的取值相同
  - 二叉树达到了限定的最大深度
    
- 获得t个iTree之后，iForest训练就结束了

---

#### 4.运用iForest判断样本是否异常

- **将训练数据x遍历每一棵iTree，然后计算h(x)、E(h(x))**
  - **h(x)：** 样本x从iTree的根节点到达叶结点所途径的路径长度，等价于样本x落入叶结点所需的划分次数
  - **E(h(x))：** 样本x在整个iForest上的平均路径长度


- **计算：c(n) = 2H(n-1) - 2(n-1)/n**
  - 其中n为训练单颗iTree的样本数，H(i)为调和级数，且H(i)=In(i)+0.577(欧拉常数)
  - c(n)用于对h(x)进行标准化
  
- **根据下列公式求异常分数**
  ![Isolation Score](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Isolation%20Forest/Pics/Isolation%20Score.jpg)

- **根据异常分数判断样本是否异常**
  - **异常分数与E(h(x))成反比，与样本异常程度成正比**
  - 当E(h(x))趋近于c(n)时，s趋近于0.5，若所有样本的异常分数均接近0.5，则表明数据中无明显异常值
  - 当E(h(x))趋近于0时，s趋近于1，此时样本x极可能是异常值
  - 当E(h(x))趋近于n-1时(即趋于最大划分次数)，s趋近于0，此时样本x极可能是正常值
  
---

#### 5.算法优势


- **缓解swamping and masking的出现**
  - **swamping**：是指将正常样本识别为异常样本；**masking**：是指异常样本没有被识别出来。这两种情况都是发生在数据量较大的情况下。
  
  - **iForest算法能有效地减缓上述两种情况发生的原因：**
    - 子采样限制了训练单颗iTree的样本数，有助于增强iTree的区分能力
    - 每一棵iTree的样本集和划分点都是随机产生的，因此每一棵iTree都具有独立性
  

- **相比基于距离或密度的算法，iForest节省了大量的计算成本**：iForest utilizes no distance or density measures to detect anomalies.This eliminates major computational cost of distance calculation in all distance-based methods and density-based methods

- **iForest的时间复杂度、内存占用较少，线性增长于样本个数**：iForest has a linear time complexity with a low
constant and a low memory requirement

- **iForest具备处理高维大数据集的能力**：iForest has the capacity to scale up to handle extremely
large data size and high-dimensional problems with a
large number of irrelevant attributes

![image](https://pic1.zhimg.com/80/v2-84c61c79358093c8833b8efc1f4d13d2_hd.jpg)
