import numpy as np


def find_center(data):
    """
    Returns the center coordinates of a two-dimensional array.

    Arguments:
        data (two-dimensional numpy.ndarray):
            The data array.

    Returns:
        numpy.ndarray: The center coordinates.
    """
    h, w = data.shape
    return np.array([h / 2, w / 2])
