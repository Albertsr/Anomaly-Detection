- **Author：** 马肖
- **E-Mail：** maxiaoscut@aliyun.com
- **GitHub：**  https://github.com/Albertsr

---

## 1. 马氏距离

#### 1.1 马氏距离等价于【规范化的主成分空间内的欧氏距离】
  
- **规范化的主成分空间**
  - 对数据集进行主成分分析，即对数据集的协方差矩阵进行特征值分解，求主成分（特征向量）
  - 对所有主成分进行归一化处理，这些规范化的主成分即构成了规范化主成分空间的坐标轴

- **将样本映射至规范化主成分空间，意味着数据从超椭圆(ellipsoidal)分布转化为超球面(spherical)分布**
  - 样本在规范化主成分空间各坐标轴上的投影(坐标分量)，可通过计算样本向量与规范化主成分的内积求得

- **两个向量的马氏距离等价于两者在规范化的主成分空间内的欧氏距离** 
  - If each of these axes is re-scaled to have unit variance, then the Mahalanobis distance corresponds to standard Euclidean distance in the transformed space. 


#### 1.2 马氏距离的特点
- **特点一：马氏距离是无单位化的、尺度无关的，它内生地考虑到了数据集各坐标轴之间的相关性**
  - The Mahalanobis distance is thus unitless and scale-invariant, and takes into account the correlations of the data set.
 
- **特点二：马氏距离与样本在各主成分上的偏离度成正比**
   - This distance is zero if P is at the mean of D, and grows as P moves away from the mean along each principal component axis

   - The Mahalanobis distance measures the number of standard deviations from P to the mean of D. 

- 参考资料：[Wikipedia : Mahalanobis distance](https://en.wikipedia.org/wiki/Mahalanobis_distance) 

---

## 2. 马氏距离的计算方法及其代码实现
#### 2.1 Python代码实现：[mahal_dist](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Mahalanobis%20Distance/mahal_dist.py) 

#### 2.2 计算样本点x距离样本集中心的马氏距离公式   
![马氏距离](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Mahalanobis%20Distance/Pics/mahal_dist.jpg)

---

## 3. 马氏距离的变体及其代码实现   
#### 3.1 Python代码实现： [mahal_dist_variant](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Mahalanobis%20Distance/mahal_dist_variant.py)

#### 3.2 论文出处： [A Novel Anomaly Detection Scheme Based on Principal Component Classifier](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Based%20on%20PCA/Papers/A%20Novel%20Anomaly%20Detection%20Scheme%20Based%20on%20Principal%20Component%20Classifier.pdf) 

#### 3.3 计算方法

  ![马氏距离变体](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Mahalanobis%20Distance/Pics/%E9%A9%AC%E6%B0%8F%E8%B7%9D%E7%A6%BB%E5%8F%98%E4%BD%93.jpg)

- **参数含义**

   ![参数含义](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Mahalanobis%20Distance/Pics/%E5%8F%98%E4%BD%93%E5%8F%82%E6%95%B0%E5%90%AB%E4%B9%89.jpg)
   
- **异常样本的判定：** 当Score(x)大于某个阈值时，便可将样本x判定为异常样本

---

## 4. 马氏距离及其变体【对样本的异常程度评估完全一致】

#### 4.1 验证方法
- 根据多个不同的随机种子生成多组实验数据集
- 根据两种方法返回的分数对样本集的索引进行升序或降序排列，例如数值最大的样本其对应的索引排在最前面，依次类推；
- 若分别根据马氏距离及其变体返回的数值大小对样本索引降序排列，若两个索引序列完全一致，则证明这两种方法对样本集中每一个样本的异常程度评估是完全一致的
- 换句话说，在数据集中随机抽取两个不同样本a与b，若马氏距离返回的数据显示样本a比样本b更偏离数据数据中心，则马氏距离变体对这种大小关系有一致的判定

#### 4.2 验证结论
- 马氏距离及其变体对**各样本在数据集中的异常程度大小关系是完全一致的**
- 马氏距离及其变体对单个样本返回的具体数值一般是不同的


#### 4.3 验证代码：[verify_mahal_equivalence](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Mahalanobis%20Distance/verify_mahal_equivalence.py)
