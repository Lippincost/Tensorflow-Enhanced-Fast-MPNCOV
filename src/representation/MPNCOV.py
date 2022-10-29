
'''
@file: MPNCOV.py
@author: Chunqiao Xu
@author: Jiangtao Xie
@author: Peihua Li
Please cite the paper below if you use the code:

Peihua Li, Jiangtao Xie, Qilong Wang and Zilin Gao. Towards Faster Training of Global Covariance Pooling Networks by Iterative Matrix Square Root Normalization. IEEE Int. Conf. on Computer Vision and Pattern Recognition (CVPR), pp. 947-955, 2018.

Peihua Li, Jiangtao Xie, Qilong Wang and Wangmeng Zuo. Is Second-order Information Helpful for Large-scale Visual Recognition? IEEE Int. Conf. on Computer Vision (ICCV),  pp. 2070-2078, 2017.

Copyright (C) 2018 Peihua Li and Jiangtao Xie

All rights reserved.
'''
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras.layers as layers


class MPNCOV(tf.keras.Model):
    """Matrix power normalized Covariance pooling (MPNCOV)
           implementation of fast MPN-COV (i.e.,iSQRT-COV)
           https://arxiv.org/abs/1712.01034

        Args:
            iterNum: #iteration of Newton-schulz method
            input_dim: the #channel of input feature
            dimension_reduction: if None, it will not use 1x1 conv to
                                  reduce the #channel of feature.
                                 if 256 or others, the #channel of feature
                                  will be reduced to 256 or others.
        """

    def __init__(self, iterNum=5, input_dim=2048, dimension_reduction=None, dropout_p=None):
        super(MPNCOV, self).__init__()
        self.iterNum = iterNum
        self.dr = dimension_reduction
        self.dropout_p = dropout_p
        self.input_dim = input_dim

        if self.dr is not None:
            if tf.keras.backend.image_data_format() == 'channels_last':
                axis = 3
            else:
                axis = 1
            self.conv_dr_block = tf.keras.Sequential(
                layers=[layers.Conv2D(self.dr, kernel_size=1, strides=1, use_bias=False, padding='same',
                                      kernel_initializer=tf.keras.initializers.VarianceScaling()),
                        layers.BatchNormalization(axis=axis,
                                                  momentum=0.9,
                                                  epsilon=1e-5,
                                                  center=True,
                                                  scale=True,
                                                  fused=True,
                                                  gamma_initializer=tf.ones_initializer()),
                        layers.ReLU()],
                name='conv_dr_block')
            output_dim = self.dr
        else:
            output_dim = input_dim
        self.output_dim = int(output_dim * (output_dim + 1) / 2)

        if self.dropout_p is not None:
            self.dropout = tf.keras.layers.Dropout(self.dropout_p)

    def call(self, x, training=None):
        if self.dr is not None:
            x = self.conv_dr_block(x, training=training)

        x = Covpool(x)
        x = Sqrtm(x)
        x = Triuvec(x)


        if self.dropout_p is not None:
            x = self.dropout(x, training=training)
        return x

@tf.custom_gradient
def Covpool(input):
    x = input
    batchSize, h, w, dim = x.shape
    M = int(h * w)
    dtype = x.dtype
    x = tf.reshape(x, [batchSize, M, dim])