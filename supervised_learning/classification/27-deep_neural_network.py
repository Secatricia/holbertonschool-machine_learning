#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or layers == []:
            raise TypeError("layers must be a list of positive integers")
        if not all(isinstance(x, int) and x > 0 for x in layers):
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if i == 0:
                self.__weights["W1"] = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.__weights["W" + str(i + 1)] = np.random.randn(layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.__weights["b" + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        self.__cache['A0'] = X
        for i in range(1, self.__L + 1):
            W = self.__weights["W" + str(i)]
            b = self.__weights["b" + str(i)]
            Z = np.matmul(W, self.__cache["A" + str(i - 1)]) + b
            self.__cache["A" + str(i)] = 1 / (1 + np.exp(-Z))
        return self.__cache["A" + str(self.__L)], self.__cache

    def cost(self, Y, A):
        m = Y.shape[1]
        cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A)) / m
        return cost

    def evaluate(self, X, Y):
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        predictions = np.argmax(A, axis=0)
        return predictions, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        m = Y.shape[1]
        dZ = cache["A" + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            A_prev = cache["A" + str(i - 1)]
            dW = np.matmul(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            dZ = np.matmul(self.__weights["W" + str(i)].T, dZ) * A_prev * (1 - A_prev)
            self.__weights["W" + str(i)] -= alpha * dW
            self.__weights["b" + str(i)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
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
        if verbose and not isinstance(step, int):
            raise TypeError("step must be an integer")
        if verbose and (step <= 0 or step > iterations):
            raise ValueError("step must be positive and <= iterations")

        costs = []
        count = []

        for i in range(iterations + 1):
            A, cache = self.forward_prop(X)
            cost = self.cost(Y, A)
            costs.append(cost)
            count.append(i)
            if verbose and (i % step == 0 or i == 0 or i == iterations):
                print("Cost after {} iterations: {}".format(i, cost))
            if i != iterations:
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(count, costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        if not filename.endswith('.pkl'):
            filename += '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        try:
            with open(filename, 'rb') as file:
                return pickle.load(file)
        except FileNotFoundError:
            return None
