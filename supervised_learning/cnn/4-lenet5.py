#!/usr/bin/env python3
"""Convolutional Neural Networks"""


import tensorflow.compat.v1 as tf

def lenet5(x, y):
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)

    conv1 = tf.keras.layers.Conv2D(
        filters=6, kernel_size=(5, 5),
        padding='same', activation=tf.nn.relu,
        kernel_initializer=initializer)(x)
    conv2 = tf.keras.layers.Conv2D(
        filters=16, kernel_size=(5, 5),
        padding='valid', activation=tf.nn.relu,
        kernel_initializer=initializer)(pool1)
    # Max pooling layer 2
    pool2 = tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2), strides=(2, 2))(conv2)

    flatten = tf.keras.layers.Flatten()(pool2)

    fc1 = tf.keras.layers.Dense(
        units=120, activation=tf.nn.relu,
        kernel_initializer=initializer)(flatten)

    fc2 = tf.keras.layers.Dense(
        units=84, activation=tf.nn.relu,
        kernel_initializer=initializer)(fc1)

    output = tf.keras.layers.Dense(
        units=10, kernel_initializer=initializer)(fc2)

    y_pred = tf.nn.softmax(output)

    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=y, logits=output))

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)

    correct_predictions = tf.equal(
        tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(
        tf.cast(correct_predictions, tf.float32))

    return y_pred, train_op, loss, accuracy
