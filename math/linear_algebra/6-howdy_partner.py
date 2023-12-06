#!/usr/bin/env python3
"""functions that concatenates two arrays"""


def cat_arrays(arr1, arr2):
    """define cat_array"""
    concat = []

    for i in range(len(arr1)):
        concat.append(arr1[i])

    for j in range(len(arr2)):
        concat.append(arr2[j])
    return concat
