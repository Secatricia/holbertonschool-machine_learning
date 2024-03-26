#!/usr/bin/env python3
"""keras"""


import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """that tests a neural network"""
    if verbose:
        return network.evaluate(data, labels, verbose=1)
    else:
        return network.evaluate(data, labels, verbose=0)
