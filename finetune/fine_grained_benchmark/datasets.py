from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from PIL import Image

import os
import os.path
import scipy.io as sio

AUTOTUNE = tf.data.experimental.AUTOTUNE

class CUB_dataset(object):
    def __init__(self, is_training=True, data_dir=None, pretrained=False, arch=None,):
        """Create  TFRecord files from Raw Images and Create an input from TFRecord files.
        Args:
          is_training: `bool` for whether the input is for training
          data_dir: `str` for the directory of the training and validation data;
              if 'null' (the literal string 'null') or implicitly False
              then construct a null pipeline, consisting of empty images
              and blank labels.
        """
        super(CUB_dataset, self).__init__()
        IMAGESIZE = 448
        self.is_training = is_training
       