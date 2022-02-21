from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import warnings

from .mpncov_resnet import *
from .resnet import *
from .vgg import *
from .mpncov_cifar_model import *
from .mpncov_vgg16bn import *

def get_basemodel(modeltype, pretraine