#!/usr/bin/env python3
"""
This module contains :
A fuction that builds the DenseNet-121 architecture

Function:
   def densenet121(growth_rate=32, compression=1.0):
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    builds the DenseNet-121 architecture
    the input data will have shape (224, 224, 3)
    """
    # Init Kernels
    init = K.initializers.VarianceScaling(scale=2.0,
                                          mode='fan_in',
                                          distribution='truncated_normal',
                                          seed=None)
    # Input datax
    inputs = K.Input(shape=(224, 224, 3))

    # Convolution 7x7
    Bn1 = K.layers.BatchNormalization()(inputs)
    relu_1 = K.layers.Activation(K.activations.relu)(Bn1)
    conv7x7 = K.layers.Conv2D(64,
                              kernel_size=(7, 7),
                              strides=2,
                              padding="same",
                              kernel_initializer=init)(relu_1)

    # Max pooling layer
    max_pool = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     padding="same",
                                     strides=2)(conv7x7)
    nb_filters = 64

    # 6 times Dense blocks
    d_block, F = dense_block(max_pool, nb_filters, growth_rate, 6)

    # Transition layer
    t_layer, Fc = transition_layer(d_block, F, compression)

    # 12 times Dense blocks
    d_block, F = dense_block(t_layer, Fc, growth_rate, 12)

    # Transition layer
    t_layer, Fc = transition_layer(d_block, F, compression)

    # 24 times Dense blocks
    d_block, F = dense_block(t_layer, Fc, growth_rate, 24)

    # Transition layer
    t_layer, Fc = transition_layer(d_block, F, compression)

    # 16 times Dense blocks
    d_block, F = dense_block(t_layer, Fc, growth_rate, 16)

    # Average Pooling
    avg_pool = K.layers.AveragePooling2D((7, 7),
                                         strides=(1, 1))(d_block)

    Dense = K.layers.Dense(1000,
                           activation='softmax',
                           kernel_initializer=init)(avg_pool)

    network = K.Model(inputs=inputs, outputs=Dense)

    return network
