- **Author：** 马肖
- **E-Mail：** maxiaoscut@aliyun.com
- **GitHub：**  https://github.com/Albertsr

---

## 1. K-邻近距离(k-distance)
   
   ![k-dist](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Local%20Outlier%20Factor/Pics/1.K-dist.jpg)

---

## 2. 可达距离(rechability distance)

  ![reach_dist](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Local%20Outlier%20Factor/Pics/2.reach_dist.jpg)

---

## 3. 局部可达密度(local reachability density)
   
   ![lrd](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Local%20Outlier%20Factor/Pics/3.lrd.jpg)
   
---

## 4. 局部离群因子(Local Outlier Factor)

   ![LOF](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Local%20Outlier%20Factor/Pics/LOF.jpg)

- LOF(A)定义为点A的第k距离邻域内各点的平均局部可达密度与点A的局部可达密度lrd的比值
- **LOF算法衡量一个数据点的异常程度，并不是看它的绝对局部密度，而是看它跟周围邻近的数据点的相对密度**
- LOF(A)越小于1，表明点A越有可能处于一个相对密集的区域，就越可能是inlier
- LOF(A)越接近于1，表明点A的局部可达与其k近邻越相似，就越不可能是异常值
- LOF(A)越大于1，表明点A越有可能与其他点较疏远，就越有可能是异常值
