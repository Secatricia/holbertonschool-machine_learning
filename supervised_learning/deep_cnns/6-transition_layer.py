#!/usr/bin/env python3
"""Deep Convolutional Architectures"""


import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """Builds a transition layer"""
    # Batch Normalization
    batch_norm = K.layers.BatchNormalization()(X)
    # ReLU activation
    relu = K.layers.Activation(K.activations.relu)(batch_norm)
    # Convolution
    conv = K.layers.Conv2D(int(nb_filters * compression),
                           kernel_size=(1, 1),
                           strides=(1, 1),
                           padding='same',
                           kernel_initializer='he_normal')(relu)
    # Average pooling
    avg_pool = K.layers.AveragePooling2D(pool_size=(2, 2),
                                          strides=(2, 2),
                                          padding='valid')(conv)

    return avg_pool, int(nb_filters * compression)
