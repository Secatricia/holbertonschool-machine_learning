#!/usr/bin/env python3
"""keras"""


import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """sets up Adam optimization for a keras model"""
    opt = K.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy'])
