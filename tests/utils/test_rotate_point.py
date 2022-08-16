import numpy as np

from gridfit.utils import rotate_point


def test_rotate_point_returns_point():
    point = np.random.rand(2)
    assert rotate_point(point, 10).shape == (2,)


def test_rotate_point_with_zero_angle_returns_point():
    point = np.random.rand(2)
    assert np.allclose(rotate_point(point, 0), point)


def test_rotate_point_rotates_around_angle():
    point = np.array([1, 1]) / np.sqrt(2)
    expected_point = np.array([0, 1])

    assert np.allclose(rotate_point(point, 45), expected_point)


def test_rotate_point_rotates_around_center():
    center = np.array([1, 1])
    point = np.array([1, 1]) / np.sqrt(2) + center
    expected_point = np.array([1, 2])

    assert np.allclose(rotate_point(point, 45, center), expected_point)
