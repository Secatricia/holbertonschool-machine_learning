#usr/bin/env python


import numpy as np
import tensorflow.compat.v1. as tf

def normalization_constants(X):

    mean = np.mean(X, axis=0)
    std_dev = np.std(X, axis=0)

    return mean, std_dev
