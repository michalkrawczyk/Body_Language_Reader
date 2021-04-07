import numpy as np


def softmax(x):
    """
    Calculates Softmax for given matrix/vector
    :param x: input matrix
    :return: matrix or vector with calculated softmax
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
