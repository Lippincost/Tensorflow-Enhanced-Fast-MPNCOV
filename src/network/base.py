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
       2) global image representaion
       3) classifier
    """
    def __init__(self, modeltype, pretrained=False):
        super(Basemodel, self).__init__()
        basemodel = get_basemodel(modeltype, pretrained)
        self.pretrained = pretrained
        if modeltype.startswith('resnet'):
            basemodel = self._reconstruct_resnet(basemodel)
        elif modeltype.startswith('mpncovresnet'):
            basemodel = self._reconstruct_mpncovresnet(basemodel)
        elif modeltype.startswith('vgg'):
            basemodel = self._recons