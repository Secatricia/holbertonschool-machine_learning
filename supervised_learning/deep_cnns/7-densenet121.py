#!/usr/bin/env python3
"""Deep Convolutional Architectures"""



import tensorflow.keras as K


def densenet121(growth_rate=32, compresion=1.0):
    """ Builds the DenseNet-121 architecture and returns the model:
        - growth_rate: growth rate,
        - compression: compression factor.
    """
    # Setting he normal and relu alias and, input shape
    hn = K.initializers.he_normal()
    relu = K.activations.relu
    ipt = K.Input(shape=(224, 224, 3))

    # Activated normalized input
    anI = K.layers.Activation(relu)(K.layers.BatchNormalization(axis=3)(ipt))

    # First layer's output
    L0 = K.layers.Conv2D(filters=64, kernel_size=(7, 7), padding='same',
                         strides=(2, 2), kernel_initializer=hn)(anI)

    # Max pooling layer's output
    MP = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                               padding='same')(L0)

    # Dense blocks and transition layers
    DB, nb_filters = dense_block(MP, 64, growth_rate, 6)
    TL, nb_filters = transition_layer(DB, nb_filters, compression)
    DB1, nb_filters = dense_block(TL, nb_filters, growth_rate, 12)
    TL1, nb_filters = transition_layer(DB1, nb_filters, compression)
    DB2, nb_filters = dense_block(TL1, nb_filters, growth_rate, 24)
    TL2, nb_filters = transition_layer(DB2, nb_filters, compression)
    DB3, nb_filters = dense_block(TL2, nb_filters, growth_rate, 16)

    # Average pooling layer's output
    AP = K.layers.AveragePooling2D(pool_size=(7, 7), strides=(1, 1),
                                   padding='valid')(DB3)

    # Dense layer's output
    DL = K.layers.Dense(1000, activation='softmax', kernel_initializer=hn)(AP)

    return K.Model(inputs=ipt, outputs=DL)
