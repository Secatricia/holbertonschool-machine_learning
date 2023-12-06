#!/usr/bin/env python3
"""Function that adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """Define add_matrices2D function"""

    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    add_matrix = []

    for i in range(len(mat1)):  # Iterate over rows
        row_sum = []
        for j in range(len(mat1[0])):  # Iterate over columns
            row_sum.append(mat1[i][j] + mat2[i][j])
        add_matrix.append(row_sum)

    return add_matrix
