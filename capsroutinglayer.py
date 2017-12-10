"""
Author: Ted Yun
"""

import numpy as np
import tensorflow as tf
from tf.keras import layers
from tf.keras.engine.topology import Layer
from tf.keras import backend as K
# import matplotlib.pyplot as plt

def squash(X, axis = None):
    X_norm = tf.norm(X, axis = axis, keep_dims = True)
    return X_norm / (1 + tf.square(x_norm)) * X

class CapsRoutingLayer(Layer):
    def __init__(self, n_output, dim_output, n_routing = 3, **kwargs):
        self.n_output = n_output
        self.dim_output = dim_output
        self.n_routing = n_routing
        super(CapsRoutingLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        """
        The input shape is (None, (number of input capsules), (dimension of input capsules))
        """
        print("Building a Capsule routing layer with input shape: ", input_shape)
        _, n_input, dim_input = input_shape
        self.n_input = n_input
        self.dim_input = dim_input
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(name = 'W', shape = (n_input, self.n_output, self.dim_output, dim_input),
            initializer = 'glorot_uniform', trainable = True)
        super(CapsRoutingLayer, self).build(input_shape)
    
    def call(self, X):
        # X.shape = (None, n_input, dim_input)
        X_reshape = K.reshape(X, (-1, self.n_input, 1, self.dim_input))
        X_tile = K.tile(X_reshape, (1, 1, self.n_output, 1))
        # X_tiled.shape = (None, n_input, n_output, dim_input)
        X_hat = K.map_fn(lambda x : K.batch_dot(self.W, x, axes = [3, 2]), elems = X_tile)
        # X_hat.shape = (None, n_input, n_output, dim_output)
        b = K.zeros(shape = (None, self.n_input, self.n_output)) # TODO: does this work?
        c = tf.nn.softmax(b, dim = 2)
        # c.shape = b.shape = (None, n_input, n_output)
        s = K.batch_dot(c, X_hat, [1, 1])
        v = squash(s)
        for iter in range(self.n_routing - 1):
            c = tf.nn.softmax(b, dim = 2)
            # c.shape = b.shape = (None, n_input, n_output)
            s = K.batch_dot(c, X_hat, [1, 1])
            v = squash(s)
            # s.shape = v.shape = (None, n_output, dim_output)
            b = b + K.batch_dot(X_hat, v, [3, 2])
            # b.shape = (None, n_input, n_output)
        return v
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_output, self.dim_output)