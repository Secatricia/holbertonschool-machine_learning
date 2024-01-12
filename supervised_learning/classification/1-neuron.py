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
