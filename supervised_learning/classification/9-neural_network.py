#!/usr/bin/env python3
"""neural network with one hidden layer performing binary classification"""


import numpy as np


class NeuralNetwork:
    """ defines a neural network with one hidden layer"""
    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__nx = nx
        self.__nodes = nodes
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """initializes the weight matrix W1 for the hidden layer
        with values randomly drawn from a normal (Gaussian) distribution."""
        return self.__W1

    @property
    def A1(self):
        """This initializes the A1 activated output for the hidden layer with 0
        This value will be updated during the forward propagation process."""
        return self.__A1

    @property
    def b1(self):
        """initializes the bias vector b1 for the hidden layer with zeros"""
        return self.__b1

    @property
    def W2(self):
        """initializes the weight matrix W1 for the hidden layer
        with values randomly drawn from a normal (Gaussian) distribution."""
        return self.__W2

    @property
    def b2(self):
        """initializes the weight matrix W1 for the hidden layer
        with values randomly drawn from a normal (Gaussian) distribution."""
        return self.__b2

    @property
    def A2(self):
        """initializes the weight matrix W1 for the hidden layer
        with values randomly drawn from a normal (Gaussian) distribution."""
        return self.__A2
