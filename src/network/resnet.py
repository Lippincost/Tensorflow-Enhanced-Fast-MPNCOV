# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains definitions for the post-activation form of Residual Networks.

Residual networks (ResNets) were proposed in:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import scipy.io as sio

__all__ = ['ResNet', 'resnet18', 'resnet26', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def conv1X1(filters, stride=1):
    """1x1 convolution"""
    return layers.Conv2D(filters, kernel_size=1, strides=stride, use_bias=False, padding='same',
                      kernel_initializer=tf.keras.initializers.VarianceScaling())

def conv3X3(filters, stride=1):
    """3x3 convolution with padding"""
    return tf.keras.Sequential([
        layers.ZeroPadding2D(padding=1),
        layers.Conv2D(filters, kernel_size=3, strides=stride, use_bias=False, padding='valid',
                      kernel_initializer=tf.keras.initializers.VarianceScaling())
    ])





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
    def __init__(self, filters, strides=1, downsample=None):
        super(residual_block, self).__init__()
        self.conv1 = conv3X3(filters=filters, stride=strides)
        self.bn1 = batch_norm(init_zero=False)
        self.relu = layers.ReLU()
        self.conv2 = conv3X3(filters=filters, stride=1)
        self.bn2 = batch_norm(init_zero=True)
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
        self.conv1 = conv1X1(filters=filters, stride=1)
        self.bn1 = batch_norm(init_zero=False)
        self.conv2 = conv3X3(filters=filters, stride=strides)
        self.bn2 = batch_norm(init_zero=False)
        self.conv3 = conv1X1(filters=filters * self.expansion, stride=1)
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


class ResNet(tf.keras.Model):
    def __init__(self, block_fn, blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_filters = 64
        self.padding = layers.ZeroPadding2D(padding=3)
        self.conv1 = layers.Conv2D(filters=64, kernel_size=7, strides=2, use_bias=False, padding='valid',
                                   kernel_initializer=tf.keras.initializers.VarianceScaling())
        self.bn1 = batch_norm(init_zero=False)
        self.relu = layers.ReLU()
        self.maxpool_padding = layers.ZeroPadding2D(padding=1)
        self.maxpool = layers.MaxPooling2D(pool_size=3, strides=2, padding='valid')

        self.layer1 = self._make_layer(block_fn, 64, blocks[0], name='stage1')
        s