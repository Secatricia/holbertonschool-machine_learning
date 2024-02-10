#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pickle

class DeepNeuralNetwork:
    """
    Class DeepNeuralNetwork : deep NN performing multiclass classification
    """

    def __init__(self, nx, layers):
        """
        class constructor

        :param nx: number of input features
        :param layers: number of nodes in each layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(n, int) and n > 0 for n in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if i == 0:
                self.__weights["W" + str(i + 1)] = (np.random.randn(layers[i], nx)
                                                    * np.sqrt(2 / nx))
            else:
                self.__weights["W" + str(i + 1)] = (np.random.randn(layers[i], layers[i - 1])
                                                    * np.sqrt(2 / layers[i - 1]))
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    def forward_prop(self, X):
        """
        method to perform forward propagation

        :param X: input data
        :return: output of the neural network
        """
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            self.__cache['A' + str(i)] = self.sigmoid(np.dot(self.__weights["W" + str(i)],
                                                              self.__cache['A' + str(i - 1)])
                                                       + self.__weights["b" + str(i)])
        return self.__cache['A' + str(self.__L)], self.__cache

    def sigmoid(self, Z):
        """
        sigmoid activation function

        :param Z: input to the activation function
        :return: output of the activation function
        """
        return 1 / (1 + np.exp(-Z))

    def cost(self, Y, A):
        """
        method to compute the cost of the neural network

        :param Y: labels of the data
        :param A: predictions by the neural network
        :return: cost
        """
        m = Y.shape[1]
        cost = -(1 / m) * np.sum(Y * np.log(A))
        return cost

    def evaluate(self, X, Y):
        """
        method to evaluate the neural network's predictions

        :param X: input data
        :param Y: labels of the data
        :return: predicted labels, cost
        """
        A, _ = self.forward_prop(X)
        predictions = np.argmax(A, axis=0)
        labels = np.argmax(Y, axis=0)
        accuracy = np.sum(predictions == labels) / Y.shape[1]
        cost = self.cost(Y, A)
        return predictions, accuracy, cost

    def train(self, X_train, Y_train, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """
        method to train the neural network

        :param X_train: input data
        :param Y_train: labels of the data
        :param iterations: number of iterations to train over
        :param alpha: learning rate
        :param verbose: flag to print training information
        :param graph: flag to display training cost graph
        :param step: frequency of printing training information
        :return: tuple containing predictions, accuracy, and cost
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")
        if not isinstance(graph, bool):
            raise TypeError("graph must be a boolean")
        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step <= 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")

        costs = []
        for i in range(iterations):
            A, _ = self.forward_prop(X_train)
            cost = self.cost(Y_train, A)
            costs.append(cost)

            if verbose and (i % step == 0 or i == iterations - 1):
                print("Cost after {} iterations: {}".format(i, cost))

            dZ = A - Y_train
            gradients = {}
            for j in range(self.__L, 0, -1):
                A_prev = self.__cache['A' + str(j - 1)]
                if j == self.__L:
                    gradients['dZ' + str(j)] = dZ
                else:
                    dZ = np.dot(self.__weights['W' + str(j + 1)].T, dZ) * (A_prev * (1 - A_prev))
                    gradients['dZ' + str(j)] = dZ
                gradients['dW' + str(j)] = np.dot(dZ, self.__cache['A' + str(j - 1)].T) / X_train.shape[1]
                gradients['db' + str(j)] = np.sum(dZ, axis=1, keepdims=True) / X_train.shape[1]

            for k in range(1, self.__L + 1):
                self.__weights['W' + str(k)] -= alpha * gradients['dW' + str(k)]
                self.__weights['b' + str(k)] -= alpha * gradients['db' + str(k)]

        if graph:
            plt.plot(range(iterations), costs)
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X_train, Y_train)

    def save(self, filename):
        """
        method to save the neural network to a file using pickle

        :param filename: name of the file
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        method to load a neural network from a file using pickle

        :param filename: name of the file
        :return: loaded neural network
        """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
