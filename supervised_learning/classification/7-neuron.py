#!/usr/bin/env python3
"""class that defines a single neuron performing binary classification"""


import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """defines Neuron class"""
    def __init__(self, nx):
        """ Define __init__ function """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.normal(size=(1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """define Getter functions for private attributes W"""
        return self.__W

    @property
    def b(self):
        """define Getter functions for private attributes b"""
        return self.__b

    @property
    def A(self):
        """define Getter functions for private attributes A"""
        return self.__A

    def forward_prop(self, X):
        """Function that Calculates the forward propagation of the neuron"""
        # Calculer la somme pondérée des entrées (z)
        z = np.dot(self.__W, X) + self.__b
        # Appliquer la fonction d'activation sigmoïde
        self.__A = 1 / (1 + np.exp(-z))
        # Renvoyer la valeur calculée
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""

        m = Y.shape[1]
        cost_function = -(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return 1 / m * np.sum(cost_function)

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        A = self.forward_prop(X)

        label_value = np.where(A >= 0.5, 1, 0)
        result_cost = self.cost(Y, A)

        return label_value, result_cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = Y.shape[1]  # Nombre d'exemples

        # Calcul du gradient par rapport aux poids
        dW = (1 / m) * np.dot(A - Y, X.T)

        # Calcul du gradient par rapport au biais
        db = (1 / m) * np.sum(A - Y)

        # Mise à jour des poids et du biais
        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """Trains the neuron"""
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")

        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")

        if alpha <= 0:
            raise ValueError("alpha must be positive")
        
        if not isinstance(step, int):
            raise TypeError("step doit être un entier")

        if step <= 0 or step > iterations:
            raise ValueError("step doit être un entier positif et <= iterations")


        Costs = []
        iterations = []

        for i in range(iterations):
            pred, cost = self.evaluate(X, Y)
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
            if i % step == 0:
                Costs.append(cost)
                iterations.append(i)
                if verbose:
                    print(f"Cost after {i} iterations: {cost}")

        if graph:
            plt.plot(iterations, Costs, 'b')
            plt.ylabel('cost')
            plt.xlabel('iteration')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)