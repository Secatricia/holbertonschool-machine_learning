#!/usr/bin/env python3
"""Deep Convolutional Architectures"""


import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block as described in Densely Connected Convolutional Networks."""
    concat_layers = [X]
    for i in range(layers):
        init = K.initializers.he_normal(seed=None)

        BN_1 = K.layers.BatchNormalization()(X)
        activation_1 = K.layers.Activation('relu')(BN_1)
        conv_1 = K.layers.Conv2D(filters=4 * growth_rate,
                                  kernel_size=(1, 1),
                                  padding='same',
                                  kernel_initializer=init)(activation_1)

        BN_2 = K.layers.BatchNormalization()(conv_1)
        activation_2 = K.layers.Activation('relu')(BN_2)
        conv_2 = K.layers.Conv2D(filters=growth_rate,
                                  kernel_size=(3, 3),
                                  padding='same',
                                  kernel_initializer=init)(activation_2)

        concat_layers.append(conv_2)
        X = K.layers.concatenate(concat_layers)
        nb_filters += growth_rate

    return X, nb_filters
