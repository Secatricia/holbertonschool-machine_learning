#!/usr/bin/env python3
"""function that concatenates two matrices along a specific axis"""


def cat_matrices2D(mat1, mat3, axis=0):
    """define cat_matrices2D function"""
    m = []
    concat_mat = []

    if axis == 0:
        for i in range(len(mat1)):
            concat_mat = []
            for j in range(len(mat1[0])):
                concat_mat.append(mat1[i][j])
            m.append(concat_mat)
        for i in range(len(mat3)):
            concat_mat = []
            for j in range(len(mat3[0])):
                concat_mat.append(mat3[i][j])
            m.append(concat_mat)

    elif axis == 1:
        if len(mat1) != len(mat3):
            return None

        for i in range(len(mat1)):
            concat_mat = mat1[i] + mat3[i]
            m.append(concat_mat)

    return m
