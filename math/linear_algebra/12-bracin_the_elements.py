#!/usr/bin/env python3
"""Function  that performs element-wise addition, subtraction, multiplication, and division"""


def np_elementwise(mat1, mat2):
    """define np_element_wise"""
    add = mat1 + mat2
    sub = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    result = [add, sub, mul, div]
    return result
