#!/usr/bin/env python3
"""Create a class Poisson that represents a poisson distribution"""


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
            self.lambtha = float(sum(data) / len(data))

            if self.lambtha <= 0:
                raise ValueError("lambtha must be a positive value")

    def pmf(self, k):
        """function that calculates the value of 
        the PMF for a given number of “successes”"""
        k = int(k)

        if k < 0:
            return 0
        else:
            result = (self.lambtha ** k) * (2.71828 ** (-self.lambtha)) / factorial(k)
            return result

    def cdf(self, k):
        """Function Calculates the value of the CDF
        for a given number of “successes”"""
        k = int(k)

        if k < 0:
            return 0
        else:
            result = 0
            pmf_sum = 0
            for i in range(k + 1):
                pmf_sum += self.pmf(i)
            result = pmf_sum
            return result


def factorial(n):
    """define factorial function"""
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)
