#!/usr/bin/env python3
"""Deep Convolutional Architectures"""


import tensorflow.keras as K

def identity_block(A_prev, filters):
    """
    Builds an identity block as described in Deep Residual Learning
    """
    F11, F3, F12 = filters

    # He normal initialization
    init = K.initializers.he_normal()

    # First component of main path
    conv1 = K.layers.Conv2D(
        filters=F11, kernel_size=(1, 1),
        strides=(1, 1), padding='valid',
        kernel_initializer=init)(A_prev)
    conv1_bn = K.layers.BatchNormalization(axis=3)(conv1)
    conv1_act = K.layers.Activation('relu')(conv1_bn)

    # Second component of main path
    conv2 = K.layers.Conv2D(
        filters=F3, kernel_size=(3, 3),
        strides=(1, 1), padding='same',
        kernel_initializer=init)(conv1_act)
    conv2_bn = K.layers.BatchNormalization(axis=3)(conv2)
    conv2_act = K.layers.Activation('relu')(conv2_bn)

    # Third component of main path
    conv3 = K.layers.Conv2D(
        filters=F12, kernel_size=(1, 1),
        strides=(1, 1), padding='valid',
        kernel_initializer=init)(conv2_act)
    conv3_bn = K.layers.BatchNormalization(axis=3)(conv3)

    # Add shortcut value to main path, and pass it through a ReLU activation
    add = K.layers.Add()([conv3_bn, A_prev])
    output = K.layers.Activation('relu')(add)

    return output
