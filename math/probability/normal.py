#!/usr/bin/env python3
"""class Normal that represents a normal distribution"""


class Normal:
    """define class Normal"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """define __init__ function"""
        if data is None:
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = sum(data) / len(data)
            sum_of_squares = sum((x - self.mean) ** 2 for x in data)
            self.stddev = (sum_of_squares / len(data)) ** 0.5

    def z_score(self, x):
        """define z_score function"""
        result = (x - self.mean) / self.stddev
        return result

    def x_value(self, z):
        """define x_value function"""
        result = z * self.stddev + self.mean
        return result

    def pdf(self, x):
        """define pdf function"""
        constant = 1 / (self.stddev * (2 * 3.1415926536) ** 0.5)
        exponent = -(x - self.mean) ** 2 / (2 * self.stddev ** 2)
        return constant * (2.7182818285 ** exponent)

    def cdf(self, x):
        """Define cdf function"""
        z = (x - self.mean) / (self.stddev * 2 ** 0.5)
        erf = (2 / 3.1415926536 ** 0.5) * (
            z - z ** 3 / 3 + z ** 5 / 10 - z ** 7 / 42 + z ** 9 / 216)
        cdf = 0.5 * (1 + erf)
        return cdf
