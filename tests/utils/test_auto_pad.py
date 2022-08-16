import numpy as np

from gridfit.utils import auto_pad


def test_auto_pad_returns_numpy_array():
    data = np.empty((5, 3))
    padded_data = auto_pad(data)

    assert isinstance(padded_data, np.ndarray)


def test_auto_pad_returns_correct_shape():
    data = np.empty((5, 3))
    padded_data = auto_pad(data)
    expected_shape = 2 * (int(np.ceil(np.sqrt(5**2 + 3**2))) // 2) + 1

    assert padded_data.shape == (expected_shape, expected_shape)


def test_auto_pad_pads_with_zeros_by_default():
    data = np.ones((5, 3))
    padded_data = auto_pad(data)

    assert padded_data[0, 0] == 0
    assert padded_data[-1, -1] == 0


def test_auto_pad_pads_with_arbitrary_value():
    data = np.ones((5, 3))
    padded_data = auto_pad(data, np.nan)

    assert np.isnan(padded_data[0, 0])
    assert np.isnan(padded_data[-1, -1])
