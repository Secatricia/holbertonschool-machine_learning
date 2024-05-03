#!/usr/bin/env python3
"""Deep Convolutional Architectures"""


import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def identity_block(A_prev, filters):
    """
    Builds an identity block as described in Deep Residual Learning
    """
    # Initialize weights
    init = K.initializers.VarianceScaling(scale=2.0, mode='fan_in',
                                          distribution='truncated_normal', seed=None)

    # Input layer
    inputs = K.Input(shape=(224, 224, 3), name='input_1')

    # Convolution 7x7
    conv7x7 = K.layers.Conv2D(64, kernel_size=(7, 7), strides=2, padding="same",
                               kernel_initializer=init, name='conv2d')(inputs)
    bn_conv7x7 = K.layers.BatchNormalization(name='batch_normalization')(conv7x7)
    relu_conv7x7 = K.layers.ReLU(name='re_lu')(bn_conv7x7)
    max_pool = K.layers.MaxPooling2D(pool_size=(3, 3), padding="same", strides=2, name='max_pooling2d')(relu_conv7x7)

    # Stage 2
    id_block_1 = projection_block(max_pool, [64, 64, 256], s=1)
    id_block_2 = identity_block(id_block_1, [64, 64, 256])
    id_block_3 = identity_block(id_block_2, [64, 64, 256])

    # Stage 3
    proj_block_1 = projection_block(id_block_3, [128, 128, 512])
    id_block_4 = identity_block(proj_block_1, [128, 128, 512])
    id_block_5 = identity_block(id_block_4, [128, 128, 512])
    id_block_6 = identity_block(id_block_5, [128, 128, 512])

    # Stage 4
    proj_block_2 = projection_block(id_block_6, [256, 256, 1024])
    id_block_7 = identity_block(proj_block_2, [256, 256, 1024])
    id_block_8 = identity_block(id_block_7, [256, 256, 1024])
    id_block_9 = identity_block(id_block_8, [256, 256, 1024])
    id_block_10 = identity_block(id_block_9, [256, 256, 1024])
    id_block_11 = identity_block(id_block_10, [256, 256, 1024])

    # Stage 5
    proj_block_3 = projection_block(id_block_11, [512, 512, 2048])
    id_block_12 = identity_block(proj_block_3, [512, 512, 2048])
    id_block_13 = identity_block(id_block_12, [512, 512, 2048])

    # Average Pooling
    avg_pool = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1), name='average_pooling2d')(id_block_13)

    # Output layer
    output = K.layers.Dense(units=1000, activation='softmax', kernel_initializer=init, name='dense')(avg_pool)

    # Define the model
    model = K.models.Model(inputs=inputs, outputs=output)

    return model
