#!/usr/bin/env python3
"""class that defines a single neuron performing binary classification"""


import numpy as np


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
        # Number of exemples
        epsilon = 1.0000001
        # Term added to avoid division by zero errors
        log_a = np.log(A + epsilon)
        log_b = np.log(1 - A + epsilon)
        cost = -(1 / m) * np.sum(Y * log_a +(1 - Y) * log_b)

        return cost
