#!/usr/bin/env python3
"""Function that concatenates two matrices along a specific axis"""


import numpy as np

def np_cat(mat1, mat2, axis=0):
    """define np_cat function"""
    if axis==0:
        mat_result = np.vstack((mat1, mat2))
        return mat_result
    elif axis == 1:
        mat_result = np.hstack((mat1, mat2))
        return mat_result
    else:
        return None
