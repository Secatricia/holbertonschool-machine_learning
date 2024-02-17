#!/usr/bin/env python3
"""placeholders"""


import tensorflow.compat.v1 as tf


def create_layer(prev, n, activation):
    """function def create_layer(prev, n, activation)"""
    initializer = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    W = tf.Variable(initializer(shape=[int(prev.shape[1]), n]), name='W')
    b = tf.Variable(tf.zeros([n]), name='b')
    z = tf.add(tf.matmul(prev, W), b)
    if activation is not None:
        return activation(z, name='layer')
    else:
        return z
