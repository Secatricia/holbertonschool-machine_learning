#!/usr/bin/env python3
"""function that determines the size of a matrix"""


def matrix_shape(mat):
    """define matrix_shape function"""
    shape = []
    while type(mat) == list:
        shape.append(len(mat))
        if mat:
            mat = mat[0]
        else:
            mat = none
    return(shape)
