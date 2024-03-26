#!/usr/bin/env python3
"""keras"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """also train the model using early stopping"""
    callbacks = []
    if validation_data and early_stopping:
        early_stop = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience)
        callbacks.append(early_stop)
    history = network.fit(
        x=data, y=labels, batch_size=batch_size, epochs=epochs,
        verbose=verbose, shuffle=shuffle, validation_data=validation_data,
        callbacks=callbacks)
    return history
