import numpy as np

from gridfit.drt import discrete_radon_transform


def test_discrete_radon_transform_returns_array(load_fixture_data):
    data = load_fixture_data('grid_test_data.npy')
    _, radon_data = discrete_radon_transform(data)

    assert isinstance(radon_data, np.ndarray)


def test_discrete_radon_transform_returns_array_with_correct_shape(load_fixture_data):  # noqa: E501
    data = load_fixture_data('grid_test_data.npy')
    angles, radon_data = discrete_radon_transform(data)
    diag = np.sqrt(data.shape[0]**2 + data.shape[1]**2)
    expected_shape = int(np.ceil(diag))

    assert radon_data.shape == (angles.shape[0], expected_shape)


def test_discrete_radon_transform_returns_radon_transform_along_chosen_axis(load_fixture_data):  # noqa: E501
    data = load_fixture_data('grid_test_data.npy')
    _, radon_data_0 = discrete_radon_transform(data, axis=0)
    _, radon_data_1 = discrete_radon_transform(data, axis=1)

    assert not np.allclose(radon_data_0, radon_data_1)


def test_discrete_radon_transform_returns_radon_transform_for_variable_steps(load_fixture_data):  # noqa: E501
    data = load_fixture_data('grid_test_data.npy')

    for steps in (10, 50, 100, 200):
        angles, _ = discrete_radon_transform(data, steps=steps)
        assert np.allclose(angles, np.linspace(-45, 45, steps))


def test_discrete_radon_transform_returns_radon_transform_for_angular_range(load_fixture_data):  # noqa: E501
    data = load_fixture_data('grid_test_data.npy')
    angles, _ = discrete_radon_transform(
        data, steps=100, angular_range=(0, 90))

    assert np.allclose(angles, np.linspace(0, 90, 100))


def test_discrete_radon_transform_does_not_pad_data_without_preprocess_flag(load_fixture_data):  # noqa: E501
    data = load_fixture_data('grid_test_data.npy')
    _, radon_data = discrete_radon_transform(data, preprocess=False)

    assert radon_data.shape[1] == data.shape[1]


def test_discrete_radon_transform_returns_expected_result(load_fixture_data):  # noqa: E501
    data = load_fixture_data('grid_test_data.npy')
    _, radon_data = discrete_radon_transform(data)
    expected_radon_data = load_fixture_data('grid_test_data_drt.npy')

    assert np.allclose(radon_data, expected_radon_data)
