#!/usr/bin/env python3
"""Deep Convolutional Architectures"""


import tensorflow.keras as K


#!/usr/bin/env python3
"""
This module contains :
Function that builds a projection block

Function:
   def projection_block(A_prev, filters, s=2):
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """builds a projection block"""

    # Init Kernel
    init = K.initializers.VarianceScaling(scale=2.0,
                                          mode='fan_in',
                                          distribution='truncated_normal',
                                          seed=None)

    # Init filters
    F11, F3, F12 = filters

    # Conv1x1
    conv1x1 = K.layers.Conv2D(F11,
                              (1, 1),
                              strides=s,
                              kernel_initializer=init)(A_prev)

    # Batch Norm
    Bn1 = K.layers.BatchNormalization()(conv1x1)

    # Relu activation
    relu_1 = K.layers.Activation(K.activations.relu)(Bn1)

    # Conv3x3
    conv3x3 = K.layers.Conv2D(F3,
                              (3, 3),
                              padding="same",
                              kernel_initializer=init)(relu_1)

    # Batch Norm
    Bn2 = K.layers.BatchNormalization()(conv3x3)

    # Relu activation
    relu_2 = K.layers.Activation(K.activations.relu)(Bn2)
    conv1x1_2 = K.layers.Conv2D(F12,
                                (1, 1),
                                kernel_initializer=init)(relu_2)

    # Batch Norm
    Bn3 = K.layers.BatchNormalization()(conv1x1_2)

    # conv1x1 shortcut connection
    conv1x1_add = K.layers.Conv2D(F12,
                                  (1, 1),
                                  strides=s,
                                  kernel_initializer=init)(A_prev)

    # Batch Norm
    Bn1_add = K.layers.BatchNormalization()(conv1x1_add)

    # Concatenate outputs
    add = K.layers.Add()([Bn3, Bn1_add])

    # Relu activation
    relu_3 = K.layers.Activation(K.activations.relu)(add)

    return relu_3
