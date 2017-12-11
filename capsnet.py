"""
Author: Ted Yun
"""

import numpy as np
import tensorflow as tf
from tf.keras import layers
from capsroutinglayer import CapsRoutingLayer
# from tf.keras import backend as K
# import matplotlib.pyplot as plt

class CapsNet():
    """
    CapsNet class
    """

    X_shape = #TODO
    n_ch1 = 256
    n_ch2 = 32
    dim_caps1 = 8
    dim_caps2 = 16
    n_kernel = 9
    reconstruction_loss_ratio = 0.00005
    n_routing = 3
    n_class = 10

    def __init__(self, X_shape, n_ch1 = 256, n_ch2 = 32, dim_caps1 = 8, dim_caps2 = 16, n_kernel = 9, n_routing = 3, reconstruction_loss_ratio = 0.00005):
        # TODO

    def build_model(self):
        X = layers.Input(shape = X_shape)
        conv1 = layers.Conv2D(filters = n_ch1, kernel_size = n_kernel, strides = (1, 1),
            padding = 'valid', activation = 'relu')(X)
        primary_caps_conv = layers.Conv2D(filters = n_ch2 * dim_caps1, kernel_size = n_kernel,
            strides = (2, 2), padding = 'valid', activation = None)(conv1)
        primary_caps = layers.Reshape(target_shape = [-1, dim_caps1])(primary_caps_conv)
        digit_caps = CapsRoutingLayer(n_output = n_class, dim_output = dim_caps2, n_routing = n_routing)(primary_caps)
    
    def train(self):
        # TODO
    
    def predict(self):
        # TODO

if __name__ == "__main__":
    # TODO