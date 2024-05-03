#!/usr/bin/env python3
"""Deep Convolutional Architectures"""


import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    input_layer = K.Input(shape=(224, 224, 3))

    # Initial Convolution layer
    conv1 = K.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same', kernel_initializer=K.initializers.he_normal(seed=None))(input_layer)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    relu1 = K.layers.Activation('relu')(bn1)
    maxpool = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(relu1)

    # Building ResNet blocks
    # Stage 1
    id1 = identity_block(maxpool, [64, 64, 256])
    id2 = identity_block(id1, [64, 64, 256])
    id3 = identity_block(id2, [64, 64, 256])

    # Stage 2
    proj1 = projection_block(id3, [128, 128, 512])
    id4 = identity_block(proj1, [128, 128, 512])
    id5 = identity_block(id4, [128, 128, 512])
    id6 = identity_block(id5, [128, 128, 512])

    # Stage 3
    proj2 = projection_block(id6, [256, 256, 1024])
    id7 = identity_block(proj2, [256, 256, 1024])
    id8 = identity_block(id7, [256, 256, 1024])
    id9 = identity_block(id8, [256, 256, 1024])
    id10 = identity_block(id9, [256, 256, 1024])
    id11 = identity_block(id10, [256, 256, 1024])

    # Stage 4
    proj3 = projection_block(id11, [512, 512, 2048])
    id12 = identity_block(proj3, [512, 512, 2048])
    id13 = identity_block(id12, [512, 512, 2048])

    # Average Pooling
    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1))(id13)

    # Flatten layer
    flatten = K.layers.Flatten()(avg_pool)

    # Output layer
    output_layer = K.layers.Dense(1000, activation='softmax')(flatten)

    # Creating model
    model = K.models.Model(inputs=input_layer, outputs=output_layer)

    return model
