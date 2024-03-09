#!/usr/bin/env python3
"""Error Analysis"""


import numpy as np


def sensitivity(confusion):
    """that calculates the precision for each class in a confusion matrix"""
    # Calculate sensitivity for each class
    true_positives = np.diag(confusion)
    false_negatives = np.sum(confusion, axis=1) - true_positives
    sensitivity_per_class = true_positives / (true_positives + false_negatives)

    return sensitivity_per_class
