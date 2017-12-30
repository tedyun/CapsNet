"""
Author: Ted Yun
"""

import numpy as np
import tensorflow as tf
from keras import layers, models, optimizers
from capslayers import CapsRoutingLayer, CapsLengthLayer
from keras import backend as K
from keras import utils as Kutils
# import matplotlib.pyplot as plt

class CapsNet():
    """
    CapsNet class
    """

    x_shape = None
    n_ch1 = 256
    n_ch2 = 32
    dim_caps1 = 8
    dim_caps2 = 16
    n_kernel = 9
    reconstruction_loss_ratio = 0.00005
    n_routing = 3
    n_class = 10
    loss_m_plus = 0.9 # m+ in margin loss
    loss_m_minus = 0.1 # m- in margin loss
    loss_lambda = 0.5

    def __init__(self, input_shape, n_ch1 = 256, n_ch2 = 32, dim_caps1 = 8, dim_caps2 = 16, n_kernel = 9, n_routing = 3, reconstruction_loss_ratio = 0.00005):
        self.x_shape = input_shape
        self.train_model = None

    def build_model(self):
        x = layers.Input(shape = self.x_shape)
        conv1 = layers.Conv2D(filters = self.n_ch1, kernel_size = self.n_kernel, strides = (1, 1),
            padding = 'valid', activation = 'relu')(x)
        primary_caps_conv = layers.Conv2D(filters = self.n_ch2 * self.dim_caps1, kernel_size = self.n_kernel,
            strides = (2, 2), padding = 'valid', activation = None)(conv1)
        primary_caps = layers.Reshape(target_shape = [-1, self.dim_caps1])(primary_caps_conv)
        digit_caps = CapsRoutingLayer(n_output = self.n_class, dim_output = self.dim_caps2, n_routing = self.n_routing)(primary_caps)
        caps_output = CapsLengthLayer()(digit_caps)
        
        y = layers.Input(shape = (self.n_class, ))

        # digit_caps.shape = (None, n_caps, dim_caps)
        print("digit_caps.shape: " + str(digit_caps.shape))
        # y.shape = (None, n_caps) as one-hot vector
        print("y.shape: " + str(y.shape))
        masked_digit_caps = self.zero_mask(digit_caps, y)

        decoder_model = self.build_decoder_model()

        train_model = models.Model([x, y], [caps_output, decoder_model(masked_digit_caps)])

        self.train_model = train_model
    
    def zero_mask(self, caps, mask):
        # caps.shape = (None, n_caps, dim_caps)
        # print("caps.shape: " + str(caps.shape))
        # mask.shape = (None, n_caps) as one-hot vector
        # print("mask.shape: " + str(mask.shape))
        return K.batch_flatten(caps * K.expand_dims(mask))
    
    def build_decoder_model(self):
        decoder_model = models.Sequential()
        decoder_model.add(layers.Dense(512, activation = 'relu', input_shape = (self.dim_caps2 * self.n_class, )))
        decoder_model.add(layers.Dense(1024, activation = 'relu'))
        decoder_model.add(layers.Dense(np.prod(self.x_shape), activation = 'sigmoid'))
        return decoder_model
    
    def margin_loss(self, y_label, y_pred_norm):
        """
        Returns the average margin loss of the batch
        """
        # loss_k.shape = (None, n_class)
        loss_k = y_label * K.square(K.maximum(0, self.loss_m_plus - y_pred_norm)) + \
            self.loss_lambda * (1 - y_label) * K.square(K.maximum(0, y_pred_norm - \
            self.loss_m_minus))
        return K.mean(K.sum(loss_k, axis = 1))
    
    def train(self, data_train, batch_size = 100, epochs = 1):
        x_train, y_train = data_train
        if self.train_model is None:
            self.build_model()
        self.train_model.compile(optimizer = optimizers.Adam(), loss = [margin_loss, 'mse'],
            loss_weights = [1, reconstruction_loss_ratio], metrics = ['accuracy'])
        self.train_model.fit([x_train, y_train], [y_train, x_train], batch_size = batch_size, epochs = epochs, validation_split = 0.1)
        return self.train_model

def load_mnist():
    from keras.datasets import mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.reshape(x_train, (-1, 28, 28, 1)).astype('float32') / 255
    x_test = np.reshape(x_test, (-1, 28, 28, 1)).astype('float32') / 255
    y_train = Kutils.to_categorical(y_train, num_classes = 10)
    y_test = Kutils.to_categorical(y_test, num_classes = 10)
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    data_train, data_test = load_mnist()
    capsnet = CapsNet(input_shape = data_train[0].shape[1:])
    capsnet.train(data_train)