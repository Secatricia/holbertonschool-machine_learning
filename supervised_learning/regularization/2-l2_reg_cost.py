#!/usr/bin/env python3
"""regularisation"""


import numpy as np


def l2_reg_cost(cost):
    """calculates the cost of a neural network with L2 regularization"""
        # Get the regularization losses
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    # Add regularization losses to the original cost
    l2_cost = cost + tf.reduce_sum(reg_losses)

    return l2_cost
