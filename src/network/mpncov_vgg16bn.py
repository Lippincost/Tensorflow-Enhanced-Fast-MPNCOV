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
                                        gamma_initializer=gamma_initializer, name=name)
class MPNCOV_VGG(tf.keras.Model):
    def __init__(self, features, classes=1000):
        super(MPNCOV_VGG, self).__init__()
        self.features = features
        self.representation = MPNCOV(input_dim=512, dimension_reduction=256, iterNum=5)
        self.classifier = tf.keras.Sequential(
            layers=[
             