#!/usr/bin/env python3
"""Deep Convolutional Architectures"""


import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Fonction pour construire le réseau Inception
    
    Returns:
    Modèle Keras représentant le réseau Inception
    """
def inception_network():
    """
    Fonction pour construire le réseau Inception
    
    Returns:
    Modèle Keras représentant le réseau Inception
    """
    inputs = K.Input(shape=(224, 224, 3), name='input_1')
    
    # Couche convolutive initiale
    conv1 = K.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv2d')(inputs)
    maxpool1 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='max_pooling2d')(conv1)
    
    # Blocs Inception
    inception1 = inception_block(maxpool1, [64, 96, 128, 16, 32, 32], name='inception_1')
    inception2 = inception_block(inception1, [128, 128, 192, 32, 96, 64], name='inception_2')
    
    # Max pooling et dropout
    maxpool2 = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='max_pooling2d_3')(inception2)
    dropout = K.layers.Dropout(0.4, name='dropout')(maxpool2)
    
    # Couche dense
    dense = K.layers.Dense(1000, activation='softmax', name='dense')(dropout)
    
    model = K.Model(inputs, dense, name='inception_network')
    
    return model

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
