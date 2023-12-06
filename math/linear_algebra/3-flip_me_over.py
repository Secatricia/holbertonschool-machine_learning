#!/usr/bin/env/python3
"""return the tranpose of 2D matrix"""


def matrix_transpose(mat):
    """
    Transpose a 2D matrix and return the transposed matrix.
    """

    nb_lignes = len(mat)
    nb_column = len(mat[0])
 
    columns = []

    for i_column in range(nb_column):
        column = [ligne[i_column] for ligne in mat]
        columns.append(column)

    return columns
