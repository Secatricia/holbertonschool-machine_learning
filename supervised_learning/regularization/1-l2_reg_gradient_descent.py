#!/usr/bin/env python3
"""regularisation"""


import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """Updates the weights and biases of a neural network"""
    m = Y.shape[1]
    # Backpropagation
    dZ = cache['A' + str(L)] - Y
    for l in range(L, 0, -1):
        A_prev = cache['A' + str(l - 1)]
        dW = np.dot(dZ, A_prev.T) / m + (lambtha / m) * weights['W' + str(l)]
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dZ = np.dot(weights['W' + str(l)].T, dZ) * (1 - np.power(A_prev, 2))
        # tanh derivative

        # Update weights and biases
        weights['W' + str(l)] -= alpha * dW
        weights['b' + str(l)] -= alpha * db
