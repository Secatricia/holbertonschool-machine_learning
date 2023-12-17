#!/usr/bin/env python3
"""class Exponential that represents an exponential distribution"""


e = 2.7182818285


class Exponential:

    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(1 / (sum(data) / len(data)))

            if self.lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
    def pdf(self, k):
        """ define pdf"""
        if k is None or k < 0:
            return 0
        return (self.lambtha * (e ** (-self.lambtha * k)))

    def cdf(self, k):
        """ define cdf"""
        if k is None or k < 0:
            return 0
        return (1 - (e ** (-self.lambtha * k)))
