#!/usr/bin/env python3
"""keras"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library"""
    model = K.Sequential()

    for i, (nodes, activation) in enumerate(zip(layers, activations)):
        if i == 0:
            model.add(K.layers.Dense(nodes, activation=activation, kernel_regularizer=K.regularizers.l2(lambtha), input_shape=(nx,)))
        else:
            model.add(K.layers.Dense(nodes, activation=activation, kernel_regularizer=K.regularizers.l2(lambtha)))
        if i < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
