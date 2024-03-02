#!/usr/bin/env python3
"""Mini-Batch"""


import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way.

    Args:
        X (numpy.ndarray): The first matrix to shuffle, of shape (m, nx).
        Y (numpy.ndarray): The second matrix to shuffle, of shape (m, ny).

    Returns:
        tuple: Shuffled X and Y matrices.
    """
    permutation = np.random.permutation(X.shape[0])
    X_shuffled = X[permutation]
    Y_shuffled = Y[permutation]
    return X_shuffled, Y_shuffled

def create_placeholders(nx, classes):
    """
    Creates placeholders for input data and labels.

    Args:
        nx (int): Number of input features.
        classes (int): Number of classes.

    Returns:
        tuple: Placeholders for input data (X) and labels (Y).
    """
    X = tf.placeholder(tf.float32, shape=[None, nx], name='X')
    Y = tf.placeholder(tf.float32, shape=[None, classes], name='Y')
    return X, Y

def forward_propagation(X, layer_sizes, activations):
    """
    Creates the forward propagation graph for the neural network.

    Args:
        X (tf.Tensor): Placeholder for input data.
        layer_sizes (list): List of layer sizes.
        activations (list): List of activation functions.

    Returns:
        tf.Tensor: Output tensor of the neural network.
    """
    A = X
    for i, size in enumerate(layer_sizes):
        A_prev = A
        if i == len(layer_sizes) - 1:
            activation = None
        else:
            activation = activations[i]
        initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
        W = tf.get_variable("W" + str(i + 1), [A_prev.shape[1], size], initializer=initializer)
        b = tf.get_variable("b" + str(i + 1), [1, size], initializer=tf.zeros_initializer())
        Z = tf.add(tf.matmul(A_prev, W), b)
        if activation is not None:
            A = activation(Z)
        else:
            A = Z
    return A

def compute_cost(Y, Y_pred):
    """
    Computes the cross-entropy cost.

    Args:
        Y (tf.Tensor): Placeholder for true labels.
        Y_pred (tf.Tensor): Predicted labels.

    Returns:
        tf.Tensor: Cost tensor.
    """
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=Y_pred, labels=Y))
    return cost

def create_minibatches(X, Y, batch_size):
    """
    Creates minibatches from the input data.

    Args:
        X (numpy.ndarray): Input data.
        Y (numpy.ndarray): Labels.
        batch_size (int): Size of each minibatch.

    Returns:
        list: List of minibatches.
    """
    m = X.shape[0]
    minibatches = []
    num_batches = m // batch_size
    for i in range(num_batches):
        start = i * batch_size
        end = start + batch_size
        minibatch_X = X[start:end]
        minibatch_Y = Y[start:end]
        minibatches.append((minibatch_X, minibatch_Y))
    if m % batch_size != 0:
        minibatch_X = X[num_batches * batch_size:]
        minibatch_Y = Y[num_batches * batch_size:]
        minibatches.append((minibatch_X, minibatch_Y))
    return minibatches

def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32, epochs=5, load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent.

    Args:
        X_train (numpy.ndarray): Training data.
        Y_train (numpy.ndarray): Training labels.
        X_valid (numpy.ndarray): Validation data.
        Y_valid (numpy.ndarray): Validation labels.
        batch_size (int): Size of each minibatch.
        epochs (int): Number of epochs for training.
        load_path (str): Path from which to load the model.
        save_path (str): Path to where the model should be saved after training.

    Returns:
        str: The path where the model was saved.
    """
    nx = X_train.shape[1]
    classes = Y_train.shape[1]
    m = X_train.shape[0]

    tf.reset_default_graph()

    # Placeholder for input data and labels
    X, Y = create_placeholders(nx, classes)

    # Forward propagation
    layer_sizes = [256, 256, 10]
    activations = [tf.nn.tanh, tf.nn.tanh, None]
    Y_pred = forward_propagation(X, layer_sizes, activations)

    # Cost function
    cost = compute_cost(Y, Y_pred)

    # Accuracy
    correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Optimization
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        # Restore the model
        saver.restore(sess, load_path)

        for epoch in range(epochs):
            X_train_shuffled, Y_train_shuffled = shuffle_data(X_train, Y_train)
            minibatches = create_minibatches(X_train_shuffled, Y_train_shuffled, batch_size)

            for i, minibatch in enumerate(minibatches):
                minibatch_X, minibatch_Y = minibatch
                _, minibatch_cost, minibatch_accuracy = sess.run([train_op, cost, accuracy], feed_dict={X: minibatch_X, Y: minibatch_Y})

                if (i + 1) % 100 == 0:
                    print("\tStep {}: ".format(i + 1))
                    print("\t\tCost: {}".format(minibatch_cost))
                    print("\t\tAccuracy: {}".format(minibatch_accuracy))

            # Print training and validation metrics
            train_cost, train_accuracy = sess.run([cost, accuracy], feed_dict={X: X_train, Y: Y_train})
            valid_cost, valid_accuracy = sess.run([cost, accuracy], feed_dict={X: X_valid, Y: Y_valid})
            print("After {} epochs:".format(epoch + 1))
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

        # Save the model
        save_path = saver.save(sess, save_path)

    return save_path

if __name__ == '__main__':
    lib = np.load('../data/MNIST.npz')
    X_train_3D = lib['X_train']
    Y_train = lib['Y_train']
    X_train = X_train_3D.reshape((X_train_3D.shape[0], -1))
