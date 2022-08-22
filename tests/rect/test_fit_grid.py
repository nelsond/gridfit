import pytest
import numpy as np

from gridfit.rect import fit_grid


def test_fit_grid_returns_array_of_correct_shape(load_fixture_data):  # noqa: E501
    data = load_fixture_data('grid_test_data_minus_50deg.npy')
    grid = fit_grid(data)

    assert grid.shape == (10, 10, 2)


def test_fit_grid_returns_full_output(load_fixture_data):  # noqa: E501
    data = load_fixture_data('grid_test_data_minus_50deg.npy')
    x, y, grid = fit_grid(data, full_output=True)

    assert grid.shape == (10, 10, 2)
    assert x.shape[0] == 10
    assert y.shape[0] == 10


def test_fit_grid_accepts_angle(load_fixture_data):  # noqa: E501
    data = load_fixture_data('grid_test_data_minus_50deg.npy')
    grid = fit_grid(data, angle=90)

    assert grid.shape == (10, 10, 2)


def test_fit_grid_accepts_debug_flag(load_fixture_data, matplotlib_figure):  # noqa: E501
    data = load_fixture_data('grid_test_data_minus_50deg.npy')
    fit_grid(data, debug=True)


def test_fit_grid_raises_value_error_for_invalid_data_type():
    for invalid_value in (10, 10., (10, 10.), [100, 100]):
        with pytest.raises(ValueError):
            fit_grid(invalid_value)


def test_fit_grid_raises_value_error_for_invalid_data_shape():
    with pytest.raises(ValueError):
        fit_grid(np.zeros((10, 10, 10)))


def test_fit_grid_raises_value_error_for_invalid_angle_type():
    for invalid_value in ('s', [10, 10]):
        with pytest.raises(ValueError):
            fit_grid(np.zeros((10, 10)), angle=invalid_value)


def test_fit_grid_warns_if_passed_data_is_not_float(load_fixture_data):
    data = load_fixture_data('grid_test_data_minus_50deg.npy').astype(int)
    with pytest.warns(UserWarning):
        fit_grid(data)