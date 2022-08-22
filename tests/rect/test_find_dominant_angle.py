import pytest
import numpy as np

from gridfit.rect import find_dominant_angle


def test_find_dominant_angle_returns_float():
    data = np.zeros((10, 10))
    assert isinstance(find_dominant_angle(data), float)


def test_find_dominant_angle_returns_expected_result(load_fixture_data):  # noqa: E501
    data = load_fixture_data('grid_test_data.npy')
    theta_1 = find_dominant_angle(data, (-90, 0))
    theta_2 = find_dominant_angle(data, (0, 90))

    assert theta_1 == pytest.approx(-50.6, abs=1e-1)
    assert theta_2 == pytest.approx(41, abs=1e-1)


def test_find_dominant_angle_accepts_debug_flag(matplotlib_figure):
    data = np.zeros((10, 10))
    find_dominant_angle(data, debug=True)


def test_find_dominant_angle_raises_value_error_for_invalid_data_type():
    for invalid_value in (10, 10., (10, 10.), [100, 100]):
        with pytest.raises(ValueError):
            find_dominant_angle(invalid_value)


def test_find_dominant_angle_raises_value_error_for_invalid_data_shape():
    with pytest.raises(ValueError):
        find_dominant_angle(np.zeros((10, 10, 10)))


def test_find_dominant_angle_raises_value_error_for_invalid_angular_range_type():
    for invalid_value in (10, [10, 10]):
        with pytest.raises(ValueError):
            find_dominant_angle(np.zeros((10, 10)), angular_range=invalid_value)

    
def test_find_dominant_angle_raises_value_error_for_invalid_angular_range_length():
    with pytest.raises(ValueError):
        find_dominant_angle(np.zeros((10, 10)), angular_range=(10, 10, 10))


def test_find_dominant_angle_raises_value_error_for_invalid_angular_range_values():
    with pytest.raises(ValueError):
        find_dominant_angle(np.zeros((10, 10)), angular_range=(10, 10))
