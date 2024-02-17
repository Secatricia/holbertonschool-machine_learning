#!/usr/bin/env python3
"""
return two placeholders
"""
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_placeholders(nx, classes):
    """
    Returns two placeholders, x and y, for the neural network
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name='x')
    y = tf.placeholder(tf.float32, shape=[None, classes], name='y')
    return x, y