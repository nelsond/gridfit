import numpy as np


def cartesian_product(x, y):
    """
    Determines the cartesian product of two one-dimensional arrays.

    Note:
        The dtype of the returned values is determined from the dtype of
        the first argument. This can have unintended side effects if the
        dtypes of the two passed arguments differ.

    Arguments:
        x (numpy.ndarray):
            The x values.

        y (numpy.ndarray):
            The y values.

    Returns:
        numpy.ndarray:
            Cartesian product of x and y returned as an array of dimension
            (n, m, 2) where n and m correspond to the length of the passed
            x and y values, respectively.
    """
    n, m = x.shape[0], y.shape[0]
    points = np.empty((n, m, 2), dtype=x.dtype)

    for i in range(n):
        for j in range(m):
            points[i, j, :] = x[i], y[j]

    return points
