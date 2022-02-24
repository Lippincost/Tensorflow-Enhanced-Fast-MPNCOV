from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import warnings

from .mpncov_resnet import *
from .resnet import *
from .vgg import *
from .mpncov_cifar_model import *
from .mpncov_vgg16bn import *

def get_basemodel(modeltype, pretrained=False):
    modeltype = globals()[modeltype]
    if pretrained == False:
        warnings.warn('You will use model that randomly initialized!')
    return modeltype(pretrained=pretrained)

class Basemodel(tf.keras.Model):
    """Load backbone model and reconstruct it into three part:
       1) feature extractor
