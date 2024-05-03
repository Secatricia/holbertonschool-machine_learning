#!/usr/bin/env python3
"""Deep Convolutional Architectures"""


import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_block(prev, filters):
    """
    Function to implement an inception block
    
    Arguments:
    prev: Tensor containing the output of the previous layer
    filters: Tuple or list of (F1, F3R, F3, F5R, F5, FPP) containing the following filters:
             - F1: number of filters for the 1x1 convolution
             - F3R: number of filters for the 1x1 convolution before the 3x3 convolution
             - F3: number of filters for the 3x3 convolution
             - F5R: number of filters for the 1x1 convolution before the 5x5 convolution
             - F5: number of filters for the 5x5 convolution
             - FPP: number of filters for the 1x1 convolution after the max pooling
             
    Returns:
    Tensor containing the concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # 1x1 convolution branch
    conv1x1 = K.layers.Conv2D(F1, (1, 1), padding='same', activation='relu')(prev)

    # 1x1 followed by 3x3 convolution branch
    conv3x3 = K.layers.Conv2D(F3R, (1, 1), padding='same', activation='relu')(prev)
    conv3x3 = K.layers.Conv2D(F3, (3, 3), padding='same', activation='relu')(conv3x3)

    # 1x1 followed by 5x5 convolution branch
    conv5x5 = K.layers.Conv2D(F5R, (1, 1), padding='same', activation='relu')(prev)
    conv5x5 = K.layers.Conv2D(F5, (5, 5), padding='same', activation='relu')(conv5x5)

    # Max pooling followed by 1x1 convolution branch
    pool = K.layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(prev)
    pool_proj = K.layers.Conv2D(FPP, (1, 1), padding='same', activation='relu')(pool)

    # Concatenate the outputs of all branches
    output = K.layers.concatenate([conv1x1, conv3x3, conv5x5, pool_proj])

    return output

def inception_network():
    """
    Function to build the Inception network
    
    Returns:
    Keras model representing the Inception network
    """
    inputs = K.Input(shape=(224, 224, 3))

    # Initial conv layer
    conv1 = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu')(inputs)
    maxpool1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(conv1)

    # Inception blocks
    inception1 = inception_block(maxpool1, (64, 96, 128, 16, 32, 32))
    inception2 = inception_block(inception1, (128, 128, 192, 32, 96, 64))

    # Max pooling and dropout
    maxpool2 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(inception2)
    dropout = K.layers.Dropout(0.4)(maxpool2)

    # Dense layer
    dense = K.layers.Dense(1000, activation='softmax')(dropout)

    model = K.Model(inputs, dense)

    return model
