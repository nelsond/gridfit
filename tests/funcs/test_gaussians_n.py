import pytest
import numpy as np

from gridfit.funcs import gaussians_n


def test_gaussians_n_returns_sum_of_gaussians():
    xx = np.arange(0, 100)
    amplitudes = np.arange(1, 10 + 1)
    np.random.shuffle(amplitudes)

    result = gaussians_n(xx, 5, 10, 0.5, 1, *amplitudes)

    for i, amplitude in enumerate(amplitudes):
        assert result[5 + i * 10] == pytest.approx(amplitude + 1)
