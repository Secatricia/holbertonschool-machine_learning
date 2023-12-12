#!/usr/bin/env python3
"""function that calculates"\sum_{i=1}^{n} i^2""""


def summation_i_squared(n):
    """define summation_i_squared fuction"""

    if not isinstance(n, int) or n < 1:
        return None

    if n == 1:
        return 1

    return n*summation_i_squared(n-1)
