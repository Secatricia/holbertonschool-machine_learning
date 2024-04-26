#!/usr/bin/env python3


import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, concatenate


def inception_block(x, filters):
    """Function to create an Inception block."""
    # 1x1 convolution branch
    branch1x1 = Conv2D(filters[0], (1, 1), padding='same', activation='relu')(x)

    # 3x3 convolution branch
    branch3x3 = Conv2D(filters[1], (1, 1), padding='same', activation='relu')(x)
    branch3x3 = Conv2D(filters[2], (3, 3), padding='same', activation='relu')(branch3x3)

    # 5x5 convolution branch
    branch5x5 = Conv2D(filters[3], (1, 1), padding='same', activation='relu')(x)
    branch5x5 = Conv2D(filters[4], (5, 5), padding='same', activation='relu')(branch5x5)

    # Max pooling branch
    branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = Conv2D(filters[5], (1, 1), padding='same', activation='relu')(branch_pool)

    # Concatenate all branches
    output = concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=3)
    return output

def inception_network():
    """Function to build the Inception network."""
    input_layer = Input(shape=(224, 224, 3))

    # First convolutional layer
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(input_layer)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Second convolutional layer
    x = Conv2D(64, (1, 1), padding='same', activation='relu')(x)
    x = Conv2D(192, (3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Inception modules
    x = inception_block(x, [64, 96, 128, 16, 32, 32])
    x = inception_block(x, [128, 128, 192, 32, 96, 64])

    # Max pooling layer
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    # Fully connected layers (not specified in the original paper)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)

    # Output layer
    output_layer = tf.keras.layers.Dense(10, activation='softmax')(x)  # Adjust output size as needed

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model

# Test the function
model = inception_network()
model.summary()
