# Tensorflow Enhanced Fast-MPNCOV

![](https://camo.githubusercontent.com/f2cdc5f25d743e922fd2c23e8a2a42e1f25c1e36/687474703a2f2f7065696875616c692e6f72672f70696374757265732f666173745f4d504e2d434f562e4a5047)
## Introduction
This repository contains the source code using the TensorFlow2.0 framework and models trained on ImageNet 2012 dataset from the following paper:<br>
```
@InProceedings{Li_2018_CVPR,
           author = {Li, Peihua and Xie, Jiangtao and Wang, Qilong and Gao, Zilin},
           title = {Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix Square Root Normalization},
           booktitle = { IEEE Int. Conf. on Computer Vision and Pattern Recognition (CVPR)},
           month = {June},
           year = {2018}
     }
```
The repository is transferred from the original author XuChunqiao to the current owner Lippincost.
It concerns an iterative matrix square root normalization network (called fast MPN-COV), which is very efficient and fit for large-scale datasets, compared to its predecessor ([MPN-COV](https://github.com/jiangtaoxie/