"""
Author: Ted Yun
"""

import numpy as np
import tensorflow as tf
from tf.keras import layers, models, optimizers
from capsroutinglayer import CapsRoutingLayer, CapsLengthLayer
from tf.keras import backend as K
# import matplotlib.pyplot as plt

class CapsNet():
    """
    CapsNet class
    """

    x_shape = #TODO
    n_ch1 = 256
    n_ch2 = 32
    dim_caps1 = 8
    dim_caps2 = 16
    n_kernel = 9
    reconstruction_loss_ratio = 0.00005
    n_routing = 3
    n_class = 10

    def __init__(self, x_shape, n_ch1 = 256, n_ch2 = 32, dim_caps1 = 8, dim_caps2 = 16, n_kernel = 9, n_routing = 3, reconstruction_loss_ratio = 0.00005):
        self.train_model = None

    def build_model(self):
        x = layers.Input(shape = x_shape)
        conv1 = layers.Conv2D(filters = n_ch1, kernel_size = n_kernel, strides = (1, 1),
            padding = 'valid', activation = 'relu')(x)
        primary_caps_conv = layers.Conv2D(filters = n_ch2 * dim_caps1, kernel_size = n_kernel,
            strides = (2, 2), padding = 'valid', activation = None)(conv1)
        primary_caps = layers.Reshape(target_shape = [-1, dim_caps1])(primary_caps_conv)
        digit_caps = CapsRoutingLayer(n_output = n_class, dim_output = dim_caps2, n_routing = n_routing)(primary_caps)
        caps_output = CapsLengthLayer()(digit_caps)
        
        y = layers.Input(shape = (n_class, ))
        masked_digit_caps = self.zero_mask(digit_caps, y)

        decoder_model = self.build_decoder_model()

        train_model = models.Model([x, y], [caps_output, decoder_model(masked_digit_caps)])

        self.train_model = train_model
    
    def margin_loss(self):
        # TODO
    
    def zero_mask(self, caps, mask):
        # caps.shape = (None, n_caps, dim_caps)
        # mask.shape = (None, n_caps) as one-hot vector
        return K.batch_flatten(caps * K.expand_dims(mask, -1))
    
    def build_decoder_model(self):
        decoder_model = models.Sequential()
        decoder_model.add(layers.Dense(512, activation = 'relu', input_shape = (dim_caps2 * n_class, )))
        decoder_model.add(layers.Dense(1024, activation = 'relu'))
        decoder_model.add(layers.Dense(K.prod(X_shape), activation = 'sigmoid'))
        return decoder_model
    
    def train(self, data_train):
        x_train, y_train = data_train
        if self.train_model is None:
            self.build_model()
        self.train_model.compile(optimizer = optimizers.Adam(), loss = [margin_loss, 'mse'],
            loss_weights = [1, reconstruction_loss_ratio], metrics = ['accuracy'])
        # TODO
        # self.train_model.fit(...)
    
    def predict(self):
        # TODO

def load_mnist:
    from keras.datasets import mnist
    return mnist.load_data()

if __name__ == "__main__":
    data_train, data_test = load_mnist()
