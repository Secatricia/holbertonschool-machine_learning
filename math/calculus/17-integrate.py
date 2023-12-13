#!/usr/bin/env python3
"""function that calculates the integral of a polynomial"""


def poly_integral(poly, C=0):
    """define poly_integral function"""
    if not isinstance(poly, list) or len(poly) < 1 or not isinstance(C, int):
        result = None
    else:
        if poly == [0]:
            result = [C]
        else:
            result = [C]
            for x in range(len(poly)):
                if poly[x] / (x + 1) == int(poly[x] / (x + 1)):
                    result.append(int(poly[x] / (x + 1)))
                else:
                    result.append(poly[x] / (x + 1))

    return result
