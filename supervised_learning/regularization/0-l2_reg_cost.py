#!/usr/bin/env python3
"""regularisation"""


import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """calculates the cost of a neural network with L2 regularization"""
    l2_reg = 0
    for i in range(1, L + 1):
        l2_reg += np.linalg.norm(weights['W' + str(i)]) ** 2

    l2_reg *= (lambtha / (2 * m))
    return cost + l2_reg