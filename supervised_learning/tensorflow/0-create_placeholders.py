#!/usr/bin/env python3
"""placeholder"""
import tensorflow.compat.v1 as tf


def create_placeholders(nx, classes):
    """
    Creates two placeholders for the intput data.
    """
    x = tf.placeholder(dtype="float32", shape=[None, nx], name="x")
    y = tf.placeholder(dtype="float32", shape=[None, classes], name="y")
    return x, y
