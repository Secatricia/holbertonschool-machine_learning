#!/usr/bin/env python3


import numpy as np


def one_hot_encode(Y, classes):
    """
    Convert a numeric label vector into a one-hot matrix.

    Args:
    Y: numpy.ndarray with shape (m,) containing numeric class labels
    classes: the maximum number of classes found in Y

    Returns:
    A one-hot encoding of Y with shape (classes, m), or None on failure
    """

    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes <= np.max(Y):
        return None

    encoded_matrix = np.zeros((classes, Y.size), dtype=float)
    encoded_matrix[Y, np.arange(Y.size)] = 1
    return encoded_matrix
