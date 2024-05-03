#!/usr/bin/env python3
"""
This module contains :
A function that builds the inception network

Function:
   def inception_network()
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds the inception network
    the input data will have shape (224, 224, 3)
    """
    # Init kernels
    init = K.initializers.HeNormal()

    # Input datas
    inputs = K.Input(shape=(224, 224, 3))

    # Conv 7x7/2
    conv7x7_2 = K.layers.Conv2D(filters=64,
                                kernel_size=(7, 7),
                                strides=(2, 2),
                                activation='relu',
                                padding='same',
                                kernel_initializer=init)(inputs)

    # Max pool 3x3/2
    max_pool = K.layers.MaxPooling2D((3, 3),
                                     strides=(2, 2),
                                     padding='same')(conv7x7_2)

    # Conv 3x3/1
    conv3x3_1 = K.layers.Conv2D(filters=64,
                                kernel_size=(1, 1),
                                activation='relu',
                                kernel_initializer=init)(max_pool)

    conv3x3_1 = K.layers.Conv2D(filters=192,
                                kernel_size=(3, 3),
                                strides=(1, 1),
                                padding="same",
                                activation='relu',
                                kernel_initializer=init)(conv3x3_1)

    # Max pool 3x3/2
    max_pool2 = K.layers.MaxPooling2D((3, 3),
                                      strides=(2, 2),
                                      padding="same")(conv3x3_1)
    # Inception bloc 3a
    i_block3a = inception_block(max_pool2, [64, 96, 128, 16, 32, 32])

    # Inception bloc 3b
    i_block3b = inception_block(i_block3a, [128, 128, 192, 32, 96, 64])

    # Max pool 3x3/2
    max_pool3 = K.layers.MaxPooling2D((3, 3),
                                      strides=(2, 2),
                                      padding="same")(i_block3b)

    # Inception bloc 4a
    i_block4a = inception_block(max_pool3, [192, 96, 208, 16, 48, 64])

    # Inception bloc 4b
    i_block4b = inception_block(i_block4a, [160, 112, 224, 24, 64, 64])

    # Inception bloc 4c
    i_block4c = inception_block(i_block4b, [128, 128, 256, 24, 64, 64])

    # Inception bloc 4d
    i_block4d = inception_block(i_block4c, [112, 144, 288, 32, 64, 64])

    # Inception bloc 4e
    i_block4e = inception_block(i_block4d, [256, 160, 320, 32, 128, 128])

    # Max pool 3x3/2
    max_pool4 = K.layers.MaxPooling2D((3, 3),
                                      strides=(2, 2),
                                      padding="same")(i_block4e)

    # Inception bloc 5a
    i_block5a = inception_block(max_pool4, [256, 160, 320, 32, 128, 128])

    # Inception bloc 5b
    i_block5b = inception_block(i_block5a, [384, 192, 384, 48, 128, 128])

    # Avg pool
    avg_pool = K.layers.AveragePooling2D((7, 7),
                                         strides=(1, 1))(i_block5b)

    # Dropout layer
    drop_out = K.layers.Dropout(0.4)(avg_pool)

    # Linear
    output = K.layers.Dense(1000,
                            activation='softmax',
                            kernel_initializer=init)(drop_out)

    network = K.Model(inputs=inputs, outputs=output)

    return network
