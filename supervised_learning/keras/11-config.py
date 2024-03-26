#!/usr/bin/env python3
"""keras"""


import tensorflow.keras as K


def save_config(network, filename):
    """model’s configuration in JSON format"""
    with open(filename, 'w') as f:
        f.write(network.to_json())

def load_config(filename):
    """loads a model with a specific configuration"""
    with open(filename, 'r') as f:
        json_config = f.read()
    model = K.models.model_from_json(json_config)
    return model
