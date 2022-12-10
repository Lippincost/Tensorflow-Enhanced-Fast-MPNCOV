"""CIFAR-10 data set.

See http://www.cs.toronto.edu/~kriz/cifar.html.
"""
import os

import tensorflow as tf

HEIGHT = 32
WIDTH = 32
DEPTH = 3


class CifarDataSet(object):
    """Cifar data set."""

    def __init__(self,
                 data_dir,
                 is_training=True,
                 pretrained=False,
                 arch=None,
                 imb_factor=None,
                 use_distortion=True):
        self.data_dir = data_dir
        self.is_training = is_training
        self.pretrained = pretrained
        self.arch = arch
        self.imb_factor = imb_factor
        self.use_distortion = use_distortion

    def get_filenames(self):

        if self.is_training