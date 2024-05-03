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
        kernel_initializer=init, name='shortcut_conv')(A_prev)
    shortcut = K.layers.BatchNormalization(
        axis=3, name='shortcut_batchnorm')(shortcut)

    # Main path
    conv1 = K.layers.Conv2D(
        filters=F11, kernel_size=(1, 1),
        strides=(s, s), padding='valid',
        kernel_initializer=init, name='conv1')(A_prev)
    conv1_bn = K.layers.BatchNormalization(
        axis=3, name='conv1_bn')(conv1)
    conv1_relu = K.layers.ReLU(name='conv1_relu')(conv1_bn)

    conv2 = K.layers.Conv2D(
        filters=F3, kernel_size=(3, 3),
        strides=(1, 1), padding='same',
        kernel_initializer=init, name='conv2')(conv1_relu)
    conv2_bn = K.layers.BatchNormalization(
        axis=3, name='conv2_bn')(conv2)
    conv2_relu = K.layers.ReLU(name='conv2_relu')(conv2_bn)

    conv3 = K.layers.Conv2D(
        filters=F12, kernel_size=(1, 1),
        strides=(1, 1), padding='valid',
        kernel_initializer=init, name='conv3')(conv2_relu)
    conv3_bn = K.layers.BatchNormalization(
        axis=3, name='conv3_bn')(conv3)

    # Add shortcut value to main path
    add = K.layers.Add(name='add')([conv3_bn, shortcut])
    output = K.layers.ReLU(name='output')(add)

    return output
