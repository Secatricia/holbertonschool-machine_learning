import numpy as np

def one_hot_encode(Y, classes):
    """
    Convert a numeric label vector into a one-hot matrix.

    Args:
    Y: numpy.ndarray with shape (m,) containing numeric class labels
    classes: the maximum number of classes found in Y

    Returns:
    A one-hot encoding of Y with shape (classes, m), or None on failure
    """
    if not isinstance(Y, np.ndarray) or Y.ndim != 1:
        return None
    
    m = Y.shape[0]
    Y_one_hot = np.zeros((classes, m))
    
    try:
        Y_one_hot[Y, np.arange(m)] = 1
        return Y_one_hot
    except IndexError:
        return None
