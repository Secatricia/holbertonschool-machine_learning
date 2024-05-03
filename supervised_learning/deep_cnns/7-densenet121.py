#!/usr/bin/env python3
"""Deep Convolutional Architectures"""


import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds the DenseNet-121 architecture
    """
    # Input layer
    X = K.Input(shape=(224, 224, 3))

    # Initial convolution layer
    init_conv = K.layers.Conv2D(
        filters=64, kernel_size=(7, 7), strides=(2, 2),
        padding='same', kernel_initializer=K.initializers.he_normal(),
        kernel_regularizer=K.regularizers.l2(1e-4))(X)
    init_bn = K.layers.BatchNormalization()(init_conv)
    init_relu = K.layers.Activation('relu')(init_bn)
    init_pool = K.layers.MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), padding='same')(init_relu)

    nb_filters = 64

    # Dense blocks and transition layers
    nb_layers = [6, 12, 24, 16]
    for i in range(4):
        dense_block_output, nb_filters = dense_block(
            init_pool, nb_filters, growth_rate, nb_layers[i])
        if i != 3:
            transition_output, nb_filters = transition_layer(
                dense_block_output, nb_filters, compression)
            init_pool = transition_output

    # Output layer
    global_avg_pool = K.layers.AveragePooling2D(
        pool_size=(7, 7), strides=(1, 1))(dense_block_output)
    dense = K.layers.Dense(
        units=1000, activation='softmax',
        kernel_initializer=K.initializers.he_normal(),
        kernel_regularizer=K.regularizers.l2(1e-4))(global_avg_pool)

    # Create model
    model = K.models.Model(inputs=X, outputs=dense)

    return model
