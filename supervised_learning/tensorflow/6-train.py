#!/usr/bin/env python3
"""placeholders"""

import tensorflow.compat.v1 as tf

# Import des fonctions des exercices précédents
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """
    builds, trains, and saves a neural network classifier
    """
    tf.set_random_seed(0)

    # Création des placeholders pour les données d'entrée et les étiquettes
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])

    # Création du graphe de propagation avant
    y_pred = forward_prop(x, layer_sizes, activations)

    # Calcul de la perte et de la précision
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    # Création de l'opération d'entraînement
    train_op = create_train_op(loss, alpha)

    # Ajout des tensors et opérations au graph
    tf.add_to_collection('placeholders', x)
    tf.add_to_collection('placeholders', y)
    tf.add_to_collection('tensors', y_pred)
    tf.add_to_collection('tensors', loss)
    tf.add_to_collection('tensors', accuracy)
    tf.add_to_collection('operation', train_op)

    # Initialisation des variables
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()

        for i in range(iterations + 1):
            # Entraînement
            _, train_cost, train_accuracy = sess.run([train_op, loss, accuracy],
                                                     feed_dict={x: X_train, y: Y_train})
            # Validation
            if i % 100 == 0 or i == 0 or i == iterations:
                valid_cost, valid_accuracy = sess.run([loss, accuracy],
                                                      feed_dict={x: X_valid, y: Y_valid})
                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_cost}")
                print(f"\tTraining Accuracy: {train_accuracy}")
                print(f"\tValidation Cost: {valid_cost}")
                print(f"\tValidation Accuracy: {valid_accuracy}")

        # Sauvegarde du modèle
        saver.save(sess, save_path)
    return save_path
