#!/usr/bin/env python3
"""
This module contains :
A function that builds the inception network

Function:
   def inception_network()
"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block



def reseau_inception():
    """
    Construit le réseau d'inception
    Les données d'entrée auront une forme (224, 224, 3)
    """
    # Initialisation des filtres
    init = K.initializers.HeNormal()

    # Définition des données d'entrée
    entrees = K.Input(shape=(224, 224, 3))

    # Convolution 7x7/2
    conv7x7_2 = K.layers.Conv2D(filters=64,
                                kernel_size=(7, 7),
                                strides=(2, 2),
                                activation='relu',
                                padding='same',
                                kernel_initializer=init)(entrees)

    # Max pooling 3x3/2
    max_pool = K.layers.MaxPooling2D((3, 3),
                                     strides=(2, 2),
                                     padding='same')(conv7x7_2)

    # Convolution 3x3/1
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

    # Max pooling 3x3/2
    max_pool2 = K.layers.MaxPooling2D((3, 3),
                                      strides=(2, 2),
                                      padding="same")(conv3x3_1)
    # Bloc Inception 3a
    bloc_inception3a = inception_block(max_pool2, [64, 96, 128, 16, 32, 32])

    # Bloc Inception 3b
    bloc_inception3b = inception_block(bloc_inception3a, [128, 128, 192, 32, 96, 64])

    # Max pooling 3x3/2
    max_pool3 = K.layers.MaxPooling2D((3, 3),
                                      strides=(2, 2),
                                      padding="same")(bloc_inception3b)

    # Bloc Inception 4a
    bloc_inception4a = inception_block(max_pool3, [192, 96, 208, 16, 48, 64])

    # Bloc Inception 4b
    bloc_inception4b = inception_block(bloc_inception4a, [160, 112, 224, 24, 64, 64])

    # Bloc Inception 4c
    bloc_inception4c = inception_block(bloc_inception4b, [128, 128, 256, 24, 64, 64])

    # Bloc Inception 4d
    bloc_inception4d = inception_block(bloc_inception4c, [112, 144, 288, 32, 64, 64])

    # Bloc Inception 4e
    bloc_inception4e = inception_block(bloc_inception4d, [256, 160, 320, 32, 128, 128])

    # Max pooling 3x3/2
    max_pool4 = K.layers.MaxPooling2D((3, 3),
                                      strides=(2, 2),
                                      padding="same")(bloc_inception4e)

    # Bloc Inception 5a
    bloc_inception5a = inception_block(max_pool4, [256, 160, 320, 32, 128, 128])

    # Bloc Inception 5b
    bloc_inception5b = inception_block(bloc_inception5a, [384, 192, 384, 48, 128, 128])

    # Pooling moyen
    pool_moyen = K.layers.AveragePooling2D((7, 7),
                                            strides=(1, 1))(bloc_inception5b)

    # Couche de dropout
    dropout = K.layers.Dropout(0.4)(pool_moyen)

    # Couche linéaire
    sortie = K.layers.Dense(1000,
                            activation='softmax',
                            kernel_initializer=init)(dropout)

    reseau = K.Model(inputs=entrees, outputs=sortie)

    return reseau
