import pytest
import numpy as np

from gridfit.utils import centroid


def test_centroid_returns_tuple_of_length_two():
    data = np.random.rand(15, 10)
    assert len(centroid(data)) == 2


def test_centroid_returns_center_for_delta_function():
    data = np.zeros((15, 10))
    data[2, 5] = 5
    com = centroid(data)

    assert com[0] == pytest.approx(2)
    assert com[1] == pytest.approx(5)


def test_centroid_returns_center_for_gaussian():
    data = np.zeros((100, 100))
    xy = np.indices(data.shape)

    data = 5 * np.exp(-((xy[0] - 40)**2 + (xy[1] - 50)**2) / (2 * 10**2))
    com = centroid(data)

    assert com[0] == pytest.approx(40, abs=1e-2)
    assert com[1] == pytest.approx(50, abs=1e-2)
