#!/usr/bin/env python3
"""Function that performs matrix multiplication"""


def mat_mul(mat1, mat2):
    """Define mat_mul function"""
    if len(mat1[0]) != len(mat2):
        return None  # Matrices cannot be multiplied

    result = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            element = 0
            for k in range(len(mat1[0])):
                element += mat1[i][k] * mat2[k][j]
            row.append(element)
        result.append(row)

    return result
