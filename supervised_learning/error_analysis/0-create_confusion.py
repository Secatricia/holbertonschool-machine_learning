#!/usr/bin/env python3
"""Error Analysis"""


import numpy as np

def create_confusion_matrix(labels, logits):
    # Convert one-hot encoded labels and logits to class indices
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)
    
    # Initialize confusion matrix
    num_classes = labels.shape[1]
    confusion = np.zeros((num_classes, num_classes))
    
    # Fill confusion matrix
    for true_class, pred_class in zip(true_classes, pred_classes):
        confusion[true_class][pred_class] += 1
    
    return confusion.astype(float)
