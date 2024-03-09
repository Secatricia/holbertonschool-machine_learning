#!/usr/bin/env python3
"""Error Analysis"""


import numpy as np


def precision(confusion):
    # Calculate precision for each class
    true_positives = np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - true_positives
    precision_per_class = true_positives / (true_positives + false_positives)
    
    return precision_per_class
