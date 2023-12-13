#!/usr/bin/env python3
"""define function that calculates the derivative of a polynomial"""


def poly_derivative(poly):
    """define poly_derivative function"""
    if isinstance(poly, list) and len(poly) == 1:
        return [0]
    if not isinstance(poly, list) or len(poly) < 2:
        return None

    derivative = []

    for i in range(1, len(poly)):
        derivative_coeff = poly[i] * i
        derivative.append(derivative_coeff)

    if not any(derivative):
        return [0]

    return derivative
