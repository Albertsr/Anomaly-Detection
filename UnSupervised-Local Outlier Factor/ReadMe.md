- **Author：** 马肖
- **E-Mail：** maxiaoscut@aliyun.com
- **GitHub：**  https://github.com/Albertsr

---

## 1. K-邻近距离(k-distance)
   
   ![k-dist](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Local%20Outlier%20Factor/Pics/1.K-dist.jpg)

---

## 2. 可达距离(rechability distance)
- 数据点A到数据点B的可达距离`$reach_{-}dist(A,B)$`为数据点`$B$`的K-邻近距离和数据点`$A,B$`之间的直接距离的最大值
  
  ![reach_dist](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Local%20Outlier%20Factor/Pics/2.reach_dist.jpg)

---

## 3. 局部可达密度(local reachability density)
   
   ![lrd](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Local%20Outlier%20Factor/Pics/3.lrd.jpg)
   
---

## 4. 局部离群因子(Local Outlier Factor)

   ![LOF](https://github.com/Albertsr/Anomaly-Detection/blob/master/UnSupervised-Local%20Outlier%20Factor/Pics/LOF.jpg)
