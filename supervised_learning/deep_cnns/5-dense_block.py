#!/usr/bin/env python3
"""Deep Convolutional Architectures"""


import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block"""
    # Initialize kernel
    init = K.initializers.VarianceScaling(scale=2.0,
                                          mode='fan_in',
                                          distribution='truncated_normal',
                                          seed=None)

    for layer in range(layers):
        # Conv 1x1
        bn = K.layers.BatchNormalization()(X)
        relu = K.layers.ReLU()(bn)
        conv = K.layers.Conv2D(filters=4 * growth_rate,
                               kernel_size=(1, 1),
                               strides=1,
                               kernel_initializer=init)(relu)

        # Conv 3x3
        bn1 = K.layers.BatchNormalization()(conv)
        relu1 = K.layers.ReLU()(bn1)
        conv1 = K.layers.Conv2D(filters=growth_rate,
                                kernel_size=(3, 3),
                                padding="same",
                                strides=1,
                                kernel_initializer=init)(relu1)

        # Concatenate
        concatenate = K.layers.Concatenate()([X, conv1])

        # Update X and filters
        X = concatenate
        nb_filters = nb_filters + growth_rate

    return concatenate, nb_filters
