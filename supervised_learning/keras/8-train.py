#!/usr/bin/env python3
"""keras"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size,
    epochs, validation_data=None, early_stopping=False,
    patience=0, learning_rate_decay=False, alpha=0.1,
    decay_rate=1, save_best=False, filepath=None,
    verbose=True, shuffle=False):
    """save the best iteration of the model"""
    if validation_data:
        if early_stopping:
            early_stop = K.callbacks.EarlyStopping(
                monitor='val_loss', patience=patience)
            callbacks.append(early_stop)
        if learning_rate_decay:
            def lr_scheduler(epoch):
                return alpha / (1 + decay_rate * epoch)
            lr_decay = K.callbacks.LearningRateScheduler(
                lr_scheduler, verbose=1)
            callbacks.append(lr_decay)
        if save_best:
            checkpoint = K.callbacks.ModelCheckpoint(
                filepath, monitor='val_loss',
                save_best_only=True, mode='min')
            callbacks.append(checkpoint)
    history = network.fit(
        x=data, y=labels, batch_size=batch_size, epochs=epochs,
        verbose=verbose, shuffle=shuffle, validation_data=validation_data,
        callbacks=callbacks)
    return history
