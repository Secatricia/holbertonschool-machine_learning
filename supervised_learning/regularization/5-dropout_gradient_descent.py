#!/usr/bin/env python3
"""regularisation"""


import numpy as np
import tensorflow.compat.v1 as tf


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """updates the weights of a neural network"""
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y
    for l in range(L, 0, -1):
        A_prev = cache['A' + str(l - 1)]
        W = weights['W' + str(l)]
        D = cache['D' + str(l)]
        dW = np.dot(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dA = np.dot(W.T, dZ)
        if l > 1:
            dA *= D
            dA /= keep_prob
        dZ = dA * (1 - np.power(A_prev, 2))  # Derivative of tanh
        weights['W' + str(l)] -= alpha * dW
        weights['b' + str(l)] -= alpha * db
