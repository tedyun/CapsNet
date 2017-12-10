"""
Author: Ted Yun
"""

import numpy as np
import tensorflow as tf
from tf.keras import layers
from tf.keras.engine.topology import Layer
# from tf.keras import backend as K
# import matplotlib.pyplot as plt

def squash(X, axis = None):
    X_norm = tf.norm(X, axis = axis, keep_dims = True)
    return X_norm / (1 + tf.square(x_norm)) * X

class CapsRoutingLayer(Layer):
    def __init__(self, n_output, dim_output, **kwargs):
        self.n_output = n_output
        self.dim_output = dim_output
        super(CapsRoutingLayer, self).__init__(**kwargs)
    def build(self, input_shape):
        """
        The input shape is (None, (number of input capsules), (dimention of input capsules))
        """
        print("Building a Capsule routing layer with input shape: ", input_shape)
        _, n_input, dim_input = input_shape
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name = 'W', shape = (n_input, self.n_output, self.dim_output, dim_input),
            initializer = 'glorot_uniform', trainable = True)
        super(CapsRoutingLayer, self).build(input_shape)
    def call(self, X):
        # TODO
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_output, self.dim_output)

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

    def __init__(self):
        # TODO

    def build_model(self):
        X = layers.Input(shape = X_shape)
        conv1 = layers.Conv2D(filters = n_ch1, kernel_size = n_kernel, strides = (1, 1),
            padding = 'valid', activation = 'relu')(X)
        primary_caps_conv = layers.Conv2D(filters = n_ch2 * dim_caps1, kernel_size = n_kernel,
            strides = (2, 2), padding = 'valid', activation = None)(conv1)
        primary_caps = layers.Reshape(target_shape = [-1, dim_caps1])(primary_caps_conv)
    
    def train(self):
        # TODO
    def predict(self):
        # TODO

if __name__ == "__main__":
    # TODO