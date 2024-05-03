#!/usr/bin/env python3
"""Deep Convolutional Architectures"""


import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """builds the ResNet-50 architecture"""
    init = K.initializers.VarianceScaling(scale=2.0,
                                          mode='fan_in',
                                          distribution='truncated_normal',
                                          seed=None)

    inputs = K.Input(shape=(224, 224, 3))

    conv7x7 = K.layers.Conv2D(64,
                              kernel_size=(7, 7),
                              strides=2,
                              padding="same",
                              kernel_initializer=init,
                              )(inputs)

    Bn1 = K.layers.BatchNormalization()(conv7x7)
    relu_1 = K.layers.ReLU()(Bn1)

    max_pool = K.layers.MaxPooling2D(pool_size=(3, 3),
                                     padding="same",
                                     strides=2)(relu_1)

    conv_block_1 = projection_block(max_pool, [64, 64, 256], s=1)
    id_block_2 = identity_block(conv_block_1, [64, 64, 256])
    id_block_3 = identity_block(id_block_2, [64, 64, 256])

    conv_block_2 = projection_block(id_block_3, [128, 128, 512])
    id_block_4 = identity_block(conv_block_2, [128, 128, 512])
    id_block_5 = identity_block(id_block_4, [128, 128, 512])
    id_block_6 = identity_block(id_block_5, [128, 128, 512])

    conv_block_3 = projection_block(id_block_6, [256, 256, 1024])
    id_block_7 = identity_block(conv_block_3, [256, 256, 1024])
    id_block_8 = identity_block(id_block_7, [256, 256, 1024])
    id_block_9 = identity_block(id_block_8, [256, 256, 1024])
    id_block_10 = identity_block(id_block_9, [256, 256, 1024])
    id_block_11 = identity_block(id_block_10, [256, 256, 1024])

    conv_block_4 = projection_block(id_block_11, [512, 512, 2048])
    id_block_12 = identity_block(conv_block_4, [512, 512, 2048])
    id_block_13 = identity_block(id_block_12, [512, 512, 2048])

    avg_pool = K.layers.AveragePooling2D((7, 7),
                                         strides=(1, 1))(id_block_13)

    Dense = K.layers.Dense(1000,
                           activation='softmax',
                           kernel_initializer=init)(avg_pool)

    network = K.Model(inputs=inputs, outputs=Dense)

    return network
