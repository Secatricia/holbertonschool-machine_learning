import numpy as np
from sensitivity import sensitivity
from precision import precision

def f1_score(confusion):
    """
    Calculate the F1 score using sensitivity and precision.
    """
    sens = sensitivity(confusion)
    prec = precision(confusion)

    f1_score_per_class = 2 * (prec * sens) / (prec + sens)

    return f1_score_per_class
