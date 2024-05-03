#!/usr/bin/env python3
"""Deep Convolutional Architectures"""


import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """Function to build the inception network"""
    # Input layer
    input_layer = K.layers.Input(shape=(224, 224, 3))

    # First Convolutional Layer
    conv1 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding='same',
                            activation='relu')(input_layer)
    maxpool1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv1)

    # Second Convolutional Layer
    conv2 = K.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same',
                            activation='relu')(maxpool1)
    conv2 = K.layers.Conv2D(filters=192, kernel_size=(3, 3), strides=(1, 1), padding='same',
                            activation='relu')(conv2)
    maxpool2 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(conv2)

    # Inception blocks
    inception3a = inception_block(maxpool2, [64, 96, 128, 16, 32, 32])
    inception3b = inception_block(inception3a, [128, 128, 192, 32, 96, 64])

    # Flatten layer
    flatten = K.layers.Flatten()(inception3b)

    # Fully connected layer
    dense1 = K.layers.Dense(units=1000, activation='relu')(flatten)

    # Output layer
    output_layer = K.layers.Dense(units=1000, activation='softmax')(dense1)

    # Define the model
    model = K.models.Model(inputs=input_layer, outputs=output_layer)

    return model
