import pytest
import numpy as np

from gridfit.roi import CircularROI


def test_initialize_sets_center():
    center = (1, 2)
    roi = CircularROI(center, 3)
    assert np.all(roi.center == center)


def test_initialize_sets_radius():
    radius = 3
    roi = CircularROI((1, 2), radius)
    assert roi.radius == radius


def test_initialize_sets_rounded_radius():
    radius = 3.5
    roi = CircularROI((1, 2), radius)
    assert roi.radius == 4


def test_mask_returns_mask_of_correct_shape():
    roi = CircularROI((10, 20), 6)

    assert np.all(np.array(roi.mask.shape) == roi.shape)


def test_apply_returns_masked_array():
    roi = CircularROI((1, 1), 1)
    data = np.arange(16).reshape((4, 4))

    assert isinstance(roi.apply(data), np.ma.MaskedArray)


def test_apply_returns_masked_array_with_correct_mask():
    roi = CircularROI((4, 4), 2)
    data = np.arange(100).reshape((10, 10))
    expected_mask = np.array([[True, True, False, True, True],
                              [True, False, False, False, True],
                              [False, False, False, False, False],
                              [True, False, False, False, True],
                              [True, True, False, True, True]])

    assert np.all(roi.apply(data).mask == expected_mask)


def test_apply_returns_masked_array_with_correct_fill_value():
    roi = CircularROI((1, 1), 1)
    data = np.arange(16).reshape((4, 4))

    assert roi.apply(data).fill_value == 0


def test_plot_returns_patch(matplotlib_figure):
    from matplotlib import patches

    roi = CircularROI((1, 1), 3)
    patch = roi.plot()

    print(type(patch))

    assert isinstance(patch, patches.Circle)


def test_plot_accepts_ax(matplotlib_figure):
    import matplotlib.pyplot as plt
    from matplotlib import patches

    ax = plt.gca()
    roi = CircularROI((1, 1), 3)
    roi.plot(ax=ax)

    assert isinstance(ax.patches[0], patches.Circle)


def test_plot_accepts_show_center_flag(matplotlib_figure):
    roi = CircularROI((1, 1), 3)
    roi.plot(show_center=True)
