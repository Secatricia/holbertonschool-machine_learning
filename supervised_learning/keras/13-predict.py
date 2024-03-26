#!/usr/bin/env python3
"""keras"""


import tensorflow.keras as K


def predict(network, data, verbose=False):
    """makes a prediction using a neural network"""
    if verbose:
        return network.predict(data, verbose=1)
    else:
        return network.predict(data)
