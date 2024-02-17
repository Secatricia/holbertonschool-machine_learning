#!/usr/bin/env python3
"""placeholders"""


import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """
    evaluates the output of a neural network
    """
    tf.reset_default_graph()

    # Load the saved model
    loader = tf.train.import_meta_graph(save_path + '.meta')

    # Placeholder names
    x_name = 'x:0'
    y_name = 'y:0'

    # Get tensors from the graph
    x = tf.get_collection('placeholders', scope=x_name)[0]
    y = tf.get_collection('placeholders', scope=y_name)[0]
    y_pred = tf.get_collection('tensors', scope='layer_2/BiasAdd:0')[0]
    loss = tf.get_collection('tensors',
                            scope='softmax_cross_entropy_loss/value:0')[0]
    accuracy = tf.get_collection('tensors', scope='Mean:0')[0]

    # Evaluate the model
    with tf.Session() as sess:
        # Restore the saved model
        loader.restore(sess, save_path)

        # Run evaluation
        pred, acc, cost = sess.run([y_pred, accuracy, loss],
                                   feed_dict={x: X, y: Y})

    return pred, acc, cost
