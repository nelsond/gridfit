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


def test_fit_grid_accepts_debug_flag(load_fixture_data):  # noqa: E501
    import matplotlib
    matplotlib.use('Agg')

    data = load_fixture_data('grid_test_data_minus_50deg.npy')
    fit_grid(data, debug=True)
