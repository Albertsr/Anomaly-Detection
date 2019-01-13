# Anomaly-Detection










## 5. 无监督异常检测算法性能对比
#### 5.1 选取的无监督异常检测算法
- Isolation Forest
- Local Outlier Factor 
- 基于KernelPCA的重构误差
- 基于LinearPCA的重构误差
- RobustPCC
- Mahalabonas Distance

#### 5.2 对比方案与数据集
- 以Isolation Forest为BaseLine
- 

#### 5.3 对比结论
- 一般来说，基于KernelPCA的重构误差**优于**基于LinearPCA的重构误差

---

## 5. 半监督异常检测算法性能对比
#### 5.1 算法
- ADOA
- PU Learning

#### 5.2 代码与对比结果
- 代码
  - [semi_contrast](https://github.com/Albertsr/Anomaly-Detection/blob/master/Algo%20Contrast/semi_contrast.py)
- 结果
  
  ![semi_contra](https://github.com/Albertsr/Anomaly-Detection/blob/master/Algo%20Contrast/semi_contra.jpg)
   

