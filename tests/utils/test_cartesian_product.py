import numpy as np

from gridfit.utils import cartesian_product


def test_cartesian_product_returns_numpy_array():
    x = np.arange(10)
    y = np.arange(3)
    prod = cartesian_product(x, y)

    assert isinstance(prod, np.ndarray)


def test_cartesian_product_returns_correct_shape():
    x = np.arange(10)
    y = np.arange(3)
    prod = cartesian_product(x, y)

    assert prod.shape == (10, 3, 2)


def test_cartesian_product_returns_cartesian_product_of_two_arrays():
    x = np.arange(10)
    y = np.arange(3)
    prod = cartesian_product(x, y)

    assert np.all(prod[0, 0] == np.array([0, 0]))
    assert np.all(prod[0, 1] == np.array([0, 1]))
    assert np.all(prod[1, 0] == np.array([1, 0]))
