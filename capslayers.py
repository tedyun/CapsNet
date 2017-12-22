"""
Author: Ted Yun
"""

import numpy as np
import tensorflow as tf
from keras import layers
from keras.engine.topology import Layer
from keras import backend as K
# import matplotlib.pyplot as plt

def squash(x, axis = None):
    """
    The "squash" function
    """
    x_norm = tf.norm(x, axis = axis, keep_dims = True)
    return x_norm / (1 + tf.square(x_norm)) * x

class CapsRoutingLayer(Layer):
    """
    A capsule-to-capsule layer using the "dynamic routing"
    """
    def __init__(self, n_output, dim_output, n_routing = 3, **kwargs):
        self.n_output = n_output
        self.dim_output = dim_output
        self.n_routing = n_routing
        self.n_input = None
        self.dim_input = None
        self.w_matrix = None
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
        self.w_matrix = self.add_weight(name = 'W', shape = (n_input, self.n_output, self.dim_output, dim_input),
            initializer = 'glorot_uniform', trainable = True)
        super(CapsRoutingLayer, self).build(input_shape)
    
    def call(self, x):
        # x.shape = (None, n_input, dim_input)
        print("CapsRoutingLayer input shape: " + str(x.shape))
        x_reshape = K.expand_dims(x, axis = 2)
        print("x_reshape.shape: " + str(x_reshape.shape))
        # x_reshape.shape = (None, n_input, 1, dim_input)
        x_tile = K.tile(x_reshape, (1, 1, self.n_output, 1))
        # x_tile.shape = (None, n_input, n_output, dim_input)
        print("x_tile.shape: " + str(x_tile.shape))
        x_hat = K.map_fn(lambda z : K.batch_dot(z, self.w_matrix, axes = [2, 3]), x_tile)
        # x_hat.shape = (None, n_input, n_output, dim_output)
        print("x_hat.shape: " + str(x_hat.shape))
        x_hat_forward_only = K.stop_gradient(x_hat)
        b = tf.zeros(shape = (tf.shape(x_hat)[0], self.n_input, self.n_output))
        v = None
        for it in range(self.n_routing):
            c = tf.nn.softmax(b, dim = 2)
            # c.shape = b.shape = (None, n_input, n_output)
            print("c.shape: " + str(c.shape))
            if it == self.n_routing - 1:
                s = K.batch_dot(c, x_hat, [[1], [1]])
                v = squash(s)
                print("s.shape: " + str(s.shape))
                # s.shape = v.shape = (None, n_output, dim_output)
            else:
                s = K.batch_dot(c, x_hat_forward_only, [1, 1])
                v = squash(s)
                # s.shape = v.shape = (None, n_output, dim_output)
                b = b + K.batch_dot(v, x_hat_forward_only, [2, 3])
                # b.shape = (None, n_input, n_output)
        print("CapsRoutingLayer output shape: " + str(v.shape))
        return v
    
    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_output, self.dim_output)

class CapsLengthLayer(Layer):
    """
    Outputs the lengths of capsule vectors. Has no weights.
    """
    def call(self, x):
        # x.shape = (None, n_input, dim_input)
        return tf.norm(x, axis = 2, keep_dims = False)
    def compute_output_shape(self, input_shape):
        return input_shape[:-1]
