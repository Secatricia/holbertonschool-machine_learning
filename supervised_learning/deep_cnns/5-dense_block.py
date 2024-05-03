#!/usr/bin/env python3
"""Deep Convolutional Architectures"""


import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block as described in Densely Connected Convolutional Networks."""
    concat_layers = [X]
    nb_filters_dense = nb_filters

    for i in range(layers):
        # Batch normalization
        X = K.layers.BatchNormalization()(X)
        # ReLU activation
        X = K.layers.Activation('relu')(X)
        # Convolution with bottleneck
        X = K.layers.Conv2D(filters=4 * growth_rate, kernel_size=1,
                            padding='same', kernel_initializer='he_normal')(X)
        # Batch normalization
        X = K.layers.BatchNormalization()(X)
        # ReLU activation
        X = K.layers.Activation('relu')(X)
        # Convolution
        X = K.layers.Conv2D(filters=growth_rate, kernel_size=3,
                            padding='same', kernel_initializer='he_normal')(X)
        # Concatenate with previous layers
        concat_layers.append(X)
        X = K.layers.concatenate(concat_layers, axis=-1)
        nb_filters_dense += growth_rate

    return X, nb_filters_dense
