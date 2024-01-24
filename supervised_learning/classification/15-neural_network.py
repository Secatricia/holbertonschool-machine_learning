#!/usr/bin/env python3
"""neural network with one hidden layer performing binary classification"""


import numpy as np
import matplotlib.pyplot as plt


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

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        Z1 = np.dot(self.__W1, X) + self.__b1
        """weighted sum of inputs (X) by weights (__W1), plus bias (__b1 )."""
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        cost = -(Y * np.log(A) + (1-Y) * np.log(1.0000001 - A))
        return 1/m * np.sum(cost)

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        A1, A2 = self.forward_prop(X)
        predicted_label = np.where(A2 >= 0.5, 1, 0)
        cost_value = self.cost(Y, A2)

        return predicted_label, cost_value

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]

        dz2 = A2 - Y
        dw2 = (1/m) * np.dot(dz2, A1.T)
        db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)

        dz1 = np.dot(self.__W2.T, dz2) * A1 * (1 - A1)
        dw1 = (1/m) * np.dot(dz1, X.T)
        db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)

        self.__W1 -= alpha * dw1
        self.__b1 -= alpha * db1
        self.__W2 -= alpha * dw2
        self.__b2 -= alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Trains the neural network"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        # Lists to store costs for plotting
        costs = []

        for i in range(iterations + 1):
            A1, A2 = self.forward_prop(X)
            cost = self.cost(Y, A2)

            # Print cost information
            if verbose and i % step == 0:
                print("Cost after {} iterations: {}".format(i, cost))

            # Store cost for plotting
            costs.append(cost)

            # Perform gradient descent
            self.gradient_descent(X, Y, A1, A2, alpha)

        # Plot the training data
        if graph:
            plt.plot(range(0, (iterations // step) * step + 1, step), costs[:iterations // step + 1], 'b-')
            plt.xlabel("Iteration")
            plt.ylabel("Cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
