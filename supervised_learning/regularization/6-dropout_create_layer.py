#!/usr/bin/env python3
"""regularisation"""


import numpy as np
import tensorflow.compat.v1 as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """creates a layer of a neural network using dropout"""
    dropout = tf.layers.Dropout(rate=1-keep_prob)
    layer = tf.layers.Dense(units=n, activation=activation)
    return dropout(layer(prev))
