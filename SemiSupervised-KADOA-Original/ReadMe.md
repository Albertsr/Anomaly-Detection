- **Author：** 马肖
- **E-Mail：** maxiaoscut@aliyun.com
- **GitHub：**  https://github.com/Albertsr


#### 1. 思路简介
- ADOA采用孤立森林与聚类相结合，KADOA运用KernelPCA重构误差替代孤立森林进行异常检测，其它思路与ADOA一致

#### 2. KADOA代码
- [KADOA.py](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-KADOA-Original/KADOA.py)

#### 3. KADOA与ADOA的性能对比
- **对比代码：** [adoa_kadoa_contrast.py](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-KADOA-Original/adoa_kadoa_contrast.py)

- **对比结果：** 在数据集、参数设置完全一致的情况下，KADOA的性能显著优于ADOA，但此结论有待更多数据集予以验证
  
  ![adoa_kadoa_contrast](https://github.com/Albertsr/Anomaly-Detection/blob/master/SemiSupervised-KADOA-Original/adoa_kadoa_contrast.jpg)
