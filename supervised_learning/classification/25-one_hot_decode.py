#!/usr/bin/env python3
"""converts a numeric label vector into a one-hot matrix"""


import numpy as np


def one_hot_decode(one_hot):
    """
    Convert a one-hot matrix into a vector of labels.

    Args:
    one_hot: a one-hot encoded numpy.ndarray with shape (classes, m)

    Returns:
    A numpy.ndarray with shape (m, )
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None

    m = one_hot.shape[1]
    decoded_labels = np.argmax(one_hot, axis=0)

    return decoded_labels
