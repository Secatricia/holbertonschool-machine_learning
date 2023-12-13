#!/usr/bin/env python3
"""function that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """define poly_integral function"""
    if not isinstance(poly, list) or not all(isinstance(coeff, (int, float)) for coeff in poly):
        return None
    
    if not isinstance(C, int):
        return None

    integral_coefficients = [C]
    for i, coeff in enumerate(poly):
        power = i + 1
        if power == 1:
            integral_coefficients.append(coeff / power)
        else:
            integral_coefficients.append(coeff / power)

    return integral_coefficients
