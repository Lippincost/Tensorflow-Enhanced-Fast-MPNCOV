
'''
resnet for cifar in pytorch

Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from ..representation.MPNCOV import *

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

__all__ = ['MPNCOV_ResNet_Cifar', 'MPNCOV_PreAct_ResNet_Cifar', 'cifar_mpncovresnet20', 'cifar_mpncovresnet32',
           'cifar_mpncovresnet44', 'cifar_mpncovresnet56', 'cifar_mpncovresnet110', 'cifar_mpncovresnet1202',
           'cifar_mpncovresnet164', 'cifar_mpncovresnet1001', 'cifar_preact_mpncovresnet110',
           'cifar_preact_mpncovresnet164', 'cifar_preact_mpncovresnet1001']

def conv1X1(filters, stride=1):
    return layers.Conv2D(filters, kernel_size=1, strides=stride, use_bias=False, padding='same')

def conv3X3(filters, stride=1):
    " 3x3 convolution with padding "
    return tf.keras.Sequential([
        layers.ZeroPadding2D(padding=1),
        layers.Conv2D(filters, kernel_size=3, strides=stride, use_bias=False, padding='valid')])

def batch_norm(init_zero=False):
    if init_zero:
        gamma_initializer = tf.zeros_initializer()
    else:
        gamma_initializer = tf.ones_initializer()

    if tf.keras.backend.image_data_format() == 'channels_last':
        axis = 3
    else:
        axis = 1
    return layers.BatchNormalization(axis=axis,
                                        momentum=BATCH_NORM_DECAY,
                                        epsilon=BATCH_NORM_EPSILON,
                                        center=True,
                                        scale=True,
                                        fused=True,
                                        gamma_initializer=gamma_initializer)
class downsample_block(tf.keras.Model):
    expansion = 1
    def __init__(self, filters, strides=1):
        super(downsample_block, self).__init__()
        self.downsample_conv = conv1X1(filters, strides)
        self.downsample_bn = batch_norm(init_zero=False)
    def call(self, x, training):
        out = self.downsample_conv(x)
        out = self.downsample_bn(out, training=training)
        return out

class residual_block(tf.keras.Model):
    expansion = 1
    def __init__(self, filters, strides=1, downsample=None):
        super(residual_block, self).__init__()
        self.conv1 = conv3X3(filters=filters, stride=strides)
        self.bn1 = batch_norm(init_zero=False)
        self.relu = layers.ReLU()
        self.conv2 = conv3X3(filters=filters)
        self.bn2 = batch_norm(init_zero=False)
        self.downsample = downsample
        self.strides = strides
    def call(self, x, training):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out, training=training)

        if self.downsample is not None:
            identity = self.downsample(x, training=training)

        out += identity
        out = self.relu(out)
        return out


class bottleneck_block(tf.keras.Model):
    expansion = 4

    def __init__(self, filters, strides=1, downsample=None):
        super(bottleneck_block, self).__init__()
        self.conv1 = conv1X1(filters=filters)
        self.bn1 = batch_norm(init_zero=False)
        self.conv2 = conv3X3(filters=filters, stride=strides)
        self.bn2 = batch_norm(init_zero=False)
        self.conv3 = conv1X1(filters=filters * self.expansion)
        self.bn3 = batch_norm(init_zero=False)
        self.relu = layers.ReLU()
        self.downsample = downsample
        self.strides = strides
    def call(self, x ,training):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out, training=training)
        out = self.relu(out)


        out = self.conv2(out)
        out = self.bn2(out, training=training)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out, training=training)
        if self.downsample is not None:
            identity = self.downsample(x, training=training)
        out += identity
        out = self.relu(out)
        return out

class PreAct_residual_block(tf.keras.Model):
    expansion = 1
    def __init__(self, filters, strides=1, downsample=None):
        super(PreAct_residual_block, self).__init__()
        self.bn1 = batch_norm(init_zero=False)
        self.relu = layers.ReLU()
        self.conv1 = conv3X3(filters=filters, stride=strides)
        self.bn2 = batch_norm(init_zero=False)
        self.conv2 = conv3X3(filters=filters)
        self.downsample = downsample
        self.strides = strides
    def call(self, x, training):
        identity = x
        out = self.bn1(x, training=training)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out, training=training)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity

        return out
class PreAct_bottleneck_block(tf.keras.Model):
    expansion = 4

    def __init__(self, filters, strides=1, downsample=None):
        super(PreAct_bottleneck_block, self).__init__()

        self.bn1 = batch_norm(init_zero=False)
        self.relu = layers.ReLU()