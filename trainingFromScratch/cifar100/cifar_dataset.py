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

        if self.is_training:
            return [os.path.join(self.data_dir, 'train' + '.tfrecords')]
        else:
            return [os.path.join(self.data_dir, 'eval' + '.tfrecords')]

    def parser(self, serialized_example):
        """Parses a single tf.Example into image and label tensors."""
        # Dimensions of the images in the CIFAR-10 dataset.
        # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of
        # the input format.
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                'image': tf.io.FixedLenFeature([], tf.string),
                'label': tf.io.FixedLenFeature([]