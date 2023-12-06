#!/usr/bin/env python3
"""function that concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat3, axis=0):
    """define cat_matrices2D function"""
    result = []
    concat_mat = []

    if axis == 0:
        for i in range(len(mat1)):
            concat_mat = []
            for j in range(len(mat1[0])):
                concat_mat.append(mat1[i][j])
            result.append(concat_mat)
        for i in range(len(mat3)):
            result.append(mat3[i])

    elif axis == 1:
        if len(mat1) != len(mat3):
            return None

        for i in range(len(mat1)):
            concat_mat = mat1[i] + mat3[i]
            result.append(concat_mat)

    return result
