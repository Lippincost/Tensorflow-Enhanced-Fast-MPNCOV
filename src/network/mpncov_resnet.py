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
from ..representation.MPNCOV import *
import scipy.io as sio

__all__ = ['MPNCOV_ResNet','mpncovresnet26','mpncovresnet50', 'mpncovresnet101']

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
        