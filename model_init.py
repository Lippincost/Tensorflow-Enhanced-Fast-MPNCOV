from src.network import *
import tensorflow as tf
import numpy as np
import warnings
__all__ = ['Newmodel', 'get_model']

class Newmodel(Basemodel):
    """replace the image representation method and classifier

       Args:
       modeltype: model archtecture
       representation: image representation method
       num_classes: the number of classes
 