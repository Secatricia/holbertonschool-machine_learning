#!/usr/bin/env python3
"""class that represents a binomial distribution"""


class Binomial:
    """define Binomial class"""
    def __init__(self, data=None, n=1, p=0.5):
        """define Binomial function"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            var = sum([(result - mean) ** 2 for result in data]) / len(data)
            self.p = 1 - (var / mean)
            self.n = round((sum(data) / self.p) / len(data))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """define pmf function"""
        k = int(k)
        if k < 0 or k > self.n:
            return 0
        a = self._calculate_probability(k)
        b = self._calculate_coefficient(k)

        return a * b

    def cdf(self, k):
        """define cdf function"""
        k = int(k)
        if k < 0:
            return 0
        cdf_value = 0
        for i in range(k + 1):
            cdf_value += self.pmf(i)
        return cdf_value

    def _calculate_coefficient(self, k):
        """define _calculate_coefficient function"""
        a = self._factorial(self.n - k) * self._factorial(k)
        b = self._factorial(self.n)
        coefficient = b / a
        return coefficient

    def _calculate_probability(self, k):
        """define _calculate_probability function"""
        probability = self.p ** k * (1 - self.p) ** (self.n - k)
        return probability

    @staticmethod
    def _factorial(n):
        """define _factorial function"""
        factorial = 1
        for i in range(1, n + 1):
            factorial = factorial*i
        return factorial
