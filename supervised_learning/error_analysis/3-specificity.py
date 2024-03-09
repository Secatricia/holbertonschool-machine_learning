#!/usr/bin/env python3
"""Error Analysis""" 


import numpy as np

def specificity(confusion):
    """that calculates the specificity for each class in a confusion matrix"""
    # Calculate specificity for each class
    tn = np.sum(confusion) - np.sum(confusion, axis=0) - np.sum(confusion, axis=1) + np.diag(confusion)
    false_positives = np.sum(confusion, axis=0) - np.diag(confusion)
    specificity_per_class = tn / (tn + false_positives)
    
    return specificity_per_class
