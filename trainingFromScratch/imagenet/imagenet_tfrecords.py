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


def _convert_to_example(filename, image_buffer, label, height, 