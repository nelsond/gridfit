import numpy as np

from gridfit.utils import find_center


def test_find_center_returns_numpy_array():
    d = np.empty((100, 99))
    assert isinstance(find_center(d), (np.ndarray))


def test_find_center_returns_center_coordinates():
    d = np.empty((100, 99))
    cy, cx = find_center(d)
    assert cy == 100 / 2 and cx == 99 / 2
