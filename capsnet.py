"""
Author: Ted Yun
"""

import numpy as np
import tensorflow as tf
from tf.keras import layers
#import matplotlib.pyplot as plt


class CapsNet():
    """
    CapsNet class
    """

    n_ch1 = 256
    n_ch2 = 32
    dim_caps1 = 8
    dim_caps2 = 16
    n_kernel = 9
    reconstruction_loss_ratio = 0.00005

    def __init__(self):
        # TODO

    def build_graph(self):
        X = layers.Input(shape = X_shape)
        conv1 = layers.Conv2D(filters = n_ch1, kernel_size = n_kernel, strides = (1, 1),
            padding = 'valid', activation = 'relu')(X)
        primary_caps_conv = layers.Conv2D(filters = dim_caps1 * n_ch2, kernel_size = n_kernel,
            strides = (2, 2), padding = 'valid', activation = None)(conv1)
        #primary_caps = layers.Reshape(target_shape = [])(primary_caps_conv)
    
    def train(self):
        # TODO
    def predict(self):
        # TODO

