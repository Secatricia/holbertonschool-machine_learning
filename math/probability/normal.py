#!/usr/bin/env python3
"""class Normal that represents a normal distribution"""


class Normal:
    """define class Normal"""
    def __init__(self, data=None, mean=0., stddev=1.):
        """define __init__ function"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            self.stddev = float((sum((x - self.mean) ** 2 for x in data) / len(data)) ** 0.5)

            if self.stddev <= 0:
                raise ValueError("stddev must be a positive value")

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
