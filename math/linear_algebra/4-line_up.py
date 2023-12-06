#!/usr/bin/env python3
"""function  that adds two arrays element-wise"""


def add_arrays(arr1, arr2):
    """function that define add_arrays"""
    if len(arr1) != len(arr2):
        return None

    add = []

    for i in range(len(arr1)):
        add.append(arr1[i]+arr2[i])
    return(add)
