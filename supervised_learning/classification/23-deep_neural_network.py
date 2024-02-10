#!/usr/bin/env python3
"""defines a deep neural network performing binary classification"""


import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        A, _ = self.forward_prop(X)
        A2 = self.__cache["A" + str(self.__L)]
        predicted_label = np.where(A2 >= 0.5, 1, 0)
        cost_value = self.cost(Y, A2)
        return predicted_label, cost_value

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]

        # Backpropagation
        dz = cache['A' + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            A = self.__cache["A" + str(i - 1)]
            dW = np.dot(dz, A.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            dz = np.dot(self.__weights["W" + str(i)].T, dz) * A * (1 - A)

            self.__weights["W" + str(i)] -= alpha * dW
            self.__weights["b" + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Trains the deep neural network"""

        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step <= 0 or step > iterations:
            raise ValueError("step must be positive and <= iterations")

        costs = []

        for iteration in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)

            if iteration % step == 0 or iteration == iterations:
                costs.append(cost)
                if verbose:
                    print("Cost after {} iterations: {}".format(iteration, cost))

            if iteration < iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph or verbose:
            plt.plot(range(0, iterations + 1, step), costs, 'b-')
            plt.xlabel('Iteration')
            plt.ylabel('Cost')
            plt.title('Training Cost')
            plt.show()

        predicted_labels, final_cost = self.evaluate(X, Y)
        print("Training completed. Final cost: {}".format(final_cost))
        return predicted_labels, final_cost
