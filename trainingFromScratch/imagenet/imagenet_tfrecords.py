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
  """Check if directory exists otherw