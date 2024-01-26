#!/usr/bin/env python3
"""defines a deep neural network performing binary classification"""


import numpy as np


class DeepNeuralNetwork:
    """defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """class constructor"""
        if not isinstance(nx, int) or nx <= 0:
            raise TypeError("nx must be a positive integer")
        if type(layers) is not list or not layers:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if type(layers[i]) is not int or layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                nodes = nx
            else:
                nodes = layers[i - 1]

            self.__weights["W" + str(i + 1)] = np.random.randn(
                layers[i], nodes) * np.sqrt(2 / nodes)

            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """Getter method for the number of layers"""
        return self.__L

    @property
    def cache(self):
        """Getter method for the cache attribute"""
        return self.__cache

    @property
    def weights(self):
        """Getter method for the weights attribute"""
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        self.__cache['A0'] = X
        A = X
        for i in range(1, self.__L + 1):
            W = self.__weights["W" + str(i)]
            b = self.__weights["b" + str(i)]
            Z = np.dot(W, A) + b
            A = 1 / (1 + np.exp(-Z))
            self.__cache["A" + str(i)] = A
        return A, self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = -(Y * np.log(A) + (1-Y) * np.log(1.0000001 - A))
        return 1/m * np.sum(cost)
