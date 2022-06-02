from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras.layers as layers
from src.representation.MPNCOV import *
import scipy.io as sio

__all__ = ['mpncov_vgg16bn']

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
def batch_norm(init_zero=False, name=None):
    if init_zero:
        gamma_initializer = tf.zeros_initializer()
    else:
        gamma_initializer = tf.ones_initializer()

 