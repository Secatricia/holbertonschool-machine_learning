#!/usr/bin/env python3
"""Convolutional Neural Networks"""


import tensorflow.compat.v1 as tf

def lenet5(x, y):
    initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
    
    # Convolutional layer 1
    conv1 = tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), padding='same', activation=tf.nn.relu,
                             kernel_initializer=initializer)(x)
    # Max pooling layer 1
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    
    # Convolutional layer 2
    conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5), padding='valid', activation=tf.nn.relu,
                             kernel_initializer=initializer)(pool1)
    # Max pooling layer 2
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv2)
    
    # Flatten the output for fully connected layers
    flatten = tf.keras.layers.Flatten()(pool2)
    
    # Fully connected layer 1
    fc1 = tf.keras.layers.Dense(units=120, activation=tf.nn.relu, kernel_initializer=initializer)(flatten)
    
    # Fully connected layer 2
    fc2 = tf.keras.layers.Dense(units=84, activation=tf.nn.relu, kernel_initializer=initializer)(fc1)
    
    # Output layer
    output = tf.keras.layers.Dense(units=10, kernel_initializer=initializer)(fc2)
    
    # Softmax activation
    y_pred = tf.nn.softmax(output)
    
    # Define loss function
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=output))
    
    # Define optimizer and training operation
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)
    
    # Calculate accuracy
    correct_predictions = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    return y_pred, train_op, loss, accuracy
