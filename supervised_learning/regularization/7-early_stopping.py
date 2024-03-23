#!/usr/bin/env python3
"""regularisation"""


import numpy as np
import tensorflow.compat.v1 as tf


def early_stopping(cost, opt_cost, threshold, patience, count):
    """determines if you should stop gradient descent early"""
    if cost > opt_cost - threshold:
        count += 1
    else:
        count = 0

    if count >= patience:
        return True, count
    else:
        return False, count
