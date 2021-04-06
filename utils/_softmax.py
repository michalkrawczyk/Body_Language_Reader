import numpy as np


def softmax(X):
    """
    Calculates Softmax for given matrix
    :param X: input matrix
    :return: matr
    """
    e_x = np.exp(X - np.max(X))
    return e_x / e_x.sum(axis=0)
