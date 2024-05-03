#!/usr/bin/env python3
"""Deep Convolutional Architectures"""


import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    Builds a projection block as described in Deep Residual Learning
    """
    F11, F3, F12 = filters

    # He normal initialization
    init = K.initializers.he_normal()

    # Shortcut path
    shortcut = K.layers.Conv2D(
        filters=F12, kernel_size=(1, 1),
        strides=(s, s), padding='valid',
        kernel_initializer=init)(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3)(shortcut)

    # Main path
    conv1 = K.layers.Conv2D(
        filters=F11, kernel_size=(1, 1),
        strides=(s, s), padding='valid',
        kernel_initializer=init)(A_prev)
    conv1_bn = K.layers.BatchNormalization(axis=3)(conv1)
    conv1_act = K.layers.Activation('relu')(conv1_bn)

    conv2 = K.layers.Conv2D(
        filters=F3, kernel_size=(3, 3),
        strides=(1, 1), padding='same',
        kernel_initializer=init)(conv1_act)
    conv2_bn = K.layers.BatchNormalization(axis=3)(conv2)
    conv2_act = K.layers.Activation('relu')(conv2_bn)

    conv3 = K.layers.Conv2D(
        filters=F12, kernel_size=(1, 1),
        strides=(1, 1), padding='valid',
        kernel_initializer=init)(conv2_act)
    conv3_bn = K.layers.BatchNormalization(axis=3)(conv3)

    # Add shortcut value to main path
    add = K.layers.Add()([conv3_bn, shortcut])
    output = K.layers.Activation('relu')(add)

    return output
