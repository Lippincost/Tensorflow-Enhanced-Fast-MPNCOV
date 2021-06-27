from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import tensorflow as tf
from PIL import Image

import os
import os.path
import scipy.io as sio

AUTOTUNE = tf.data.experimental.AUTOTUNE