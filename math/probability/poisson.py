#!/usr/bin/env python3
"""Create a class Poisson that represents a poisson distribution"""


e = 2.7182818285

class Poisson:
    """define class"""
    def __init__(self, data=None, lambtha=1.):
        """define function __init__ """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """Calculates the pmf function"""
        if k is not int:
            k = int(k)
        if k < 0:
            return 0
        factoriel = 1
        for i in range(1, k + 1):
            factoriel = factoriel * i
        return (e ** -self.lambtha * self.lambtha ** k) / factoriel

    def cdf(self, k):
        """Calculates the cdf function"""
        if k is not int:
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(0, k + 1):
            factoriel = 1
            for j in range(1, i + 1):
                factoriel = factoriel * j
            cdf += self.lambtha ** i / factoriel
        return cdf * e ** -self.lambtha
