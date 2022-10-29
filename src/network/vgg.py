
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import tensorflow.keras.layers as layers

__all__ = ['VGG','vgg16', 'vgg19']
class VGG(tf.keras.Model):
    def __init__(self, features, classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = tf.keras.Sequential(
            layers=[
                layers.Flatten(),
                layers.Dense(4096, activation='relu'),
                layers.Dense(4096, activation='relu'),
                layers.Dense(classes)],
            name='classifier')
    def call(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x