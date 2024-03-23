#!/usr/bin/env python3
"""regularisation"""


import numpy as np
import tensorflow.compat.v1 as tf


def dropout_forward_prop(X, weights, L, keep_prob):
    """conducts forward propagation using Dropout"""
    cache = {}
    cache['A0'] = X
    for l in range(1, L + 1):
        A_prev = cache['A' + str(l - 1)]
        W = weights['W' + str(l)]
        b = weights['b' + str(l)]
        Z = np.dot(W, A_prev) + b
        if l < L:
            A = np.tanh(Z)
            # Apply dropout
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A *= D
            A /= keep_prob
            cache['D' + str(l)] = D
        else:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
        cache['Z' + str(l)] = Z
        cache['A' + str(l)] = A
    return cache
