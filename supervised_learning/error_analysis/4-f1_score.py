#!/usr/bin/env python3
"""Error Analysis""" 


import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ calculates the F1 score of a confusion matrix"""
    classes = confusion.shape[0]
    f1_scores = np.zeros(classes)

    for i in range(classes):
        s_i = sensitivity(confusion, i)
        precision_i = precision(confusion, i)
        if s_i + precision_i == 0:
            f1_scores[i] = 0
        else:
            f1_scores[i] = 2 * (s_i * precision_i) / (s_i + precision_i)

    return f1_scores
