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
    if not isinstance(Y, np.ndarray) or Y.ndim != 1:
            return None

        if not isinstance(classes, int) or classes < 2 or classes < Y.max():
            return None

        one_hot = np.eye(classes)[Y]

        return one_hot.T
