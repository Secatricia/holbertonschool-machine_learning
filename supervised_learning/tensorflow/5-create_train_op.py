#!/usr/bin/env python3
"""placeholders"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """creates the training operation for the network"""
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    return train_op
