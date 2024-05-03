#!/usr/bin/env python3
"""Deep Convolutional Architectures"""


import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """Builds a dense block as described in Densely Connected Convolutional Networks."""
    # He normal initialization
    initializer = K.initializers.he_normal(seed=None)

    # Save the input tensor for concatenation
    concat_list = [X]

    # Iterate through the number of layers
    for i in range(layers):
        # Compute the number of filters
        filters = nb_filters + i * growth_rate

        # Bottleneck layer
        X = K.layers.BatchNormalization(axis=3)(X)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(filters * 4, kernel_size=1, padding='same', kernel_initializer=initializer)(X)

        # 3x3 convolutional layer
        X = K.layers.BatchNormalization(axis=3)(X)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(filters, kernel_size=3, padding='same', kernel_initializer=initializer)(X)

        # Append the output of the layer to the concatenation list
        concat_list.append(X)

        # Concatenate the list
        X = K.layers.concatenate(concat_list)

    # Return the concatenated output and the number of filters
    return X, filters
