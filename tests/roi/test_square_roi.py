import pytest
import numpy as np

from gridfit.roi import SquareROI


def test_initialize_sets_center():
    center = (1, 2)
    roi = SquareROI(center, 3)
    assert np.all(roi.center == center)


def test_initialize_sets_size():
    size = 3
    roi = SquareROI((1, 2), size)
    assert np.all(roi.size == size)


def test_initialize_raises_error_for_invalid_center():
    with pytest.raises(ValueError):
        SquareROI((1, 2, 3), 4)

    with pytest.raises(ValueError):
        SquareROI((1, 'test'), 4)

    with pytest.raises(ValueError):
        SquareROI(('test', 1), 4)

    with pytest.raises(ValueError):
        SquareROI('test', 4)


def test_initialize_raises_error_for_invalid_size():
    with pytest.raises(ValueError):
        SquareROI((1, 2), 0)

    with pytest.raises(ValueError):
        SquareROI((1, 2), -1)

    with pytest.raises(ValueError):
        SquareROI((1, 2), 'test')


def test_center_rounded_rounds_center_coordinates():
    roi = SquareROI((5.1, 8.5), 4)
    expected_center_rounded = (5, 8)

    assert np.all(roi.center_rounded == expected_center_rounded)


def test_size_rounded_rounds_size():
    roi = SquareROI((1, 2), 8.5)
    expected_size = 8

    assert roi.size_rounded == expected_size


def test_boundaries_rounds_center():
    roi = SquareROI((5.1, 8.5), 4)
    expected_boundaries = ((5 - 4 // 2, 8 - 4 // 2),
                           (5 + 4 // 2, 8 + 4 // 2))

    assert np.all(roi.boundaries == np.array(expected_boundaries))


def test_boundaries_rounds_size():
    roi = SquareROI((5, 8), 4.5)
    expected_boundaries = ((5 - 4 // 2, 8 - 4 // 2),
                           (5 + 4 // 2, 8 + 4 // 2))

    assert np.all(roi.boundaries == np.array(expected_boundaries))


def test_boundaries_with_even_size():
    roi = SquareROI((5, 8), 4)
    expected_boundaries = ((5 - 4 // 2, 8 - 4 // 2),
                           (5 + 4 // 2, 8 + 4 // 2))

    assert np.all(roi.boundaries == np.array(expected_boundaries))


def test_boundaries_with_uneven_size():
    roi = SquareROI((5, 8), 5)
    expected_boundaries = ((5 - 5 // 2, 8 - 5 // 2),
                           (5 + 5 // 2 + 1, 8 + 5 // 2 + 1))

    assert np.all(roi.boundaries == np.array(expected_boundaries))


def test_boundaries_with_negative_values():
    roi = SquareROI((1, 2), 8)
    expected_boundaries = ((1 - 8 // 2, 2 - 8 // 2),
                           (1 + 8 // 2, 2 + 8 // 2))

    assert np.all(roi.boundaries == np.array(expected_boundaries))


def test_shape_returns_size_rounded():
    roi = SquareROI((1, 2), 8.5)
    expected_shape = (8, 8)

    assert roi.shape == expected_shape


def test_apply_raises_error_for_invalid_data():
    roi = SquareROI((1, 2), 8.5)

    with pytest.raises(ValueError):
        roi.apply('test')

    with pytest.raises(ValueError):
        roi.apply(np.array([1, 2, 3]))


def test_apply_raises_error_if_roi_outside_data():
    data = np.zeros((3, 3))

    roi = SquareROI((1, 2), 8.5)
    with pytest.raises(ValueError):
        roi.apply(data)

    roi = SquareROI((5, 5), 1)
    with pytest.raises(ValueError):
        roi.apply(data)


def test_apply_returns_roi_data():
    roi = SquareROI((1, 1), 3)
    data = np.arange(16).reshape((4, 4))

    expected_data = np.array([[0, 1, 2],
                              [4, 5, 6],
                              [8, 9, 10]])

    assert np.all(roi.apply(data) == expected_data)


def test_apply_returns_view():
    roi = SquareROI((1, 1), 3)
    data = np.arange(16).reshape((4, 4))
    roi_data = roi.apply(data)
    data[1, 1] = 100
    expected_data = np.array([[0, 1, 2],
                              [4, 100, 6],
                              [8, 9, 10]])

    assert np.all(roi_data == expected_data)


def test_plot_returns_rectangle_patch(matplotlib_figure):
    from matplotlib import patches

    roi = SquareROI((1, 1), 3)
    patch = roi.plot()

    assert isinstance(patch, patches.Rectangle)


def test_plot_accepts_ax(matplotlib_figure):
    import matplotlib.pyplot as plt
    from matplotlib import patches

    ax = plt.gca()
    roi = SquareROI((1, 1), 3)
    roi.plot(ax=ax)

    assert isinstance(ax.patches[0], patches.Rectangle)


def test_plot_accepts_show_center_flag(matplotlib_figure):
    roi = SquareROI((1, 1), 3)
    roi.plot(show_center=True)
