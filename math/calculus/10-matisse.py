#!/usr/bin/env python3
"""define function that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """define poly_derivative function"""
    if not isinstance(poly, list):
        return None

    if not all(isinstance(coeff, (int, float)) for coeff in poly):
        return None

    if not poly or poly == [0]:
        return [0]

    derivative = []

    for i in range(1, len(poly)):
        derivative_coeff = poly[i] * i
        derivative.append(derivative_coeff)

    if not any(derivative):
        return [0]

    return derivative
