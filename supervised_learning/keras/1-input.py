#!/usr/bin/env python3
"""keras"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """builds a neural network with the Keras library"""
    inputs = K.Input(shape=(nx,))
    x = inputs
    for i, (nodes, activation) in enumerate(zip(layers, activations)):
        kernel_reg = K.regularizers.l2(lambtha) if lambtha else None
        if i != len(layers) - 1:
            x = K.layers.Dense(
                nodes,
                activation=activation, 
                kernel_regularizer=kernel_reg)(x)
            x = K.layers.Dropout(1 - keep_prob)(x)
        else:
            x = K.layers.Dense(nodes, activation=activation, kernel_regularizer=kernel_reg)(x)
    model = K.Model(inputs=inputs, outputs=x)
    return model
