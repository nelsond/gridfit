import pytest
import numpy as np

from gridfit.utils import rms_size


def test_rms_size_returns_tuple_of_length_two():
    data = np.random.rand(10, 10)
    assert len(rms_size(data)) == 2


def test_rms_size_returns_width_for_gaussian():
    data = np.zeros((100, 100))
    xy = np.indices(data.shape)

    data = 10 * np.exp(-((xy[0] - 40)**2 + (xy[1] - 50)**2) / (2 * 10**2))
    rms = rms_size(data)

    assert rms[0] == pytest.approx(10, abs=1e-2)
    assert rms[1] == pytest.approx(10, abs=1e-2)
