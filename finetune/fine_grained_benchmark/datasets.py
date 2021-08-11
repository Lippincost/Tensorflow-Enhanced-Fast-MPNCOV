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
        self.data_dir = data_dir
        self.pretrained = pretrained
        self.arch = arch

        def preprocess_train_image_randomflip(image_bytes):
            shape = tf.image.extract_jpeg_shape(image_bytes)
            image_height = shape[0]
            image_width = shape[1]

            padded_center_crop_size = tf.cast(tf.minimum(image_height, image_width), tf.int32)

            offset_height = ((image_height - padded_center_crop_size) + 1) // 2
            offset_width = ((image_width - padded_center_crop_size) + 1) // 2
            crop_window = tf.stack([offset_height, offset_width,
                                    padded_center_crop_size, padded_center_crop_size])
            image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
            image = tf.image.resize(image, [IMAGESIZE, IMAGESIZE])
            image = tf.image.random_flip_left_right(image)
            if self.pretrained and self.arch.startswith('vgg'):
                # RGB==>BGR for VGG16
                image = image[..., ::-1]
                mean = [0.406 * 255, 0.456 * 255, 0.485 * 255]
                std = None
            else:
                mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
                std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
            if mean is not None:
                image = tf.subtract(image, mean)
       