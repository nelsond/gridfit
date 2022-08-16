import numpy as np

from gridfit.utils import rotate


def test_rotate_returns_array_of_same_shape():
    data = np.zeros((10, 10))
    assert rotate(data, 0).shape == data.shape


def test_rotate_rotates_by_zero_degrees():
    data = np.random.rand(10, 10)
    data[2, 5] = 5
    rotated = rotate(data, 0)

    assert np.any(data == rotated)


def test_rotate_does_not_change_dtype():
    for dtype in (np.float32, np.uint16):
        data = np.zeros((10, 10), dtype=dtype)
        assert rotate(data, 20).dtype == dtype


def test_rotate_rotates_by_positive_angle(load_fixture_data):
    data = load_fixture_data('grid_test_data.npy')
    expected_data = load_fixture_data('grid_test_data_plus_40deg.npy')

    rotated = rotate(data, 40)

    assert np.allclose(rotated, expected_data)


def test_rotate_rotates_by_negative_angle(load_fixture_data):
    data = load_fixture_data('grid_test_data.npy')
    expected_data = load_fixture_data('grid_test_data_minus_50deg.npy')

    rotated = rotate(data, -50)

    assert np.allclose(rotated, expected_data)
