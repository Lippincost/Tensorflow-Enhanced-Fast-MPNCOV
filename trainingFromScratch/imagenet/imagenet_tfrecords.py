"""
Script to convert dataset .

"""

import math
import os
import random
import tensorflow as tf
import argparse




TRAINING_SHARDS = 1000
VALIDATION_SHARDS = 100

TRAINING_DIRECTORY = 'train'
VALIDATION_DIRECTORY = 'validation'

parser = argparse.ArgumentParser(description='Convert imagenet dataset to TFRECORDS')
parser.add_argument('--dataset', metavar='DIR', default=None,
                    help='path to dataset')


def _check_or_create_dir(directory):
  """Check if directory exists otherwise create it."""
  if not os.path.exists(directory):
    os.makedirs(directory)


def _int64_feature(value):
  """Wrapper for inserting int64 features into Example proto."""
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
  """Wrapper for inserting bytes features into Example proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _convert_to_example(filename, image_buffer, label, height, width):
  """Build an Example proto for an example.

  Args:
    filename: string, path to an image file, e.g., '/path/to/example.JPG'
    image_buffer: string, JPEG encoding of RGB image
    label: integer, identifier for the ground truth for the network
    height: integer, image height in pixels
    width: integer, image width in pixels
  Returns:
    Example proto
  """
  image_format = b'JPEG'

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': _int64_feature(height),
      'image/width': _int64_feature(width),
      'image/class/label': _int64_feature(label),
      'image/format': _bytes_feature(image_format),
      'image/encoded': _bytes_feature(image_buffer)}))
  return example


def _is_png(filename):
  """Determine if a file contains a PNG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a PNG.
  """
  # File list from:
  # https://github.com/cytsai/ilsvrc-cmyk-image-list
  return 'n02105855_2933.JPEG' in filename


def _is_cmyk(filename):
  """Determine if file contains a CMYK JPEG format image.

  Args:
    filename: string, path of the image file.

  Returns:
    boolean indicating if the image is a JPEG encoded with CMYK color space.
  """
  # File list from:
  # https://github.com/cytsai/ilsvrc-cmyk-image-list
  blacklist = set(['n01739381_1309.JPEG', 'n02077923_14822.JPEG',
                   'n02447366_23489.JPEG', 'n02492035_15739.JPEG',
                   'n02747177_10752.JPEG', 'n03018349_4028.JPEG',
                   'n03062245_4620.JPEG', 'n03347037_9675.JPEG',
                   'n03467068_12171.JPEG', 'n03529860_11437.JPEG',
                   'n03544143_17228.JPEG', 'n03633091_5218.JPEG',
                   'n03710637_5125.JPEG', 'n03961711_5286.JPEG',
                   'n04033995_2932.JPEG', 'n04258138_17003.JPEG',
                   'n04264628_27969.JPEG', 'n04336792_7448.JPEG',
                   'n04371774_5854.JPEG', 'n04596742_4225.JPEG',
                   'n07583066_647.JPEG', 'n13037406_4650.JPEG'])
  return os.path.basename(filename) in blacklist


class ImageCoder(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
      return

  def png_to_jpeg(self, image_data):
      image = tf.image.decode_png(image_data, ch