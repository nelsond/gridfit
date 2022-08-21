import sys
import os

import pytest

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_path, '..'))


@pytest.fixture
def load_fixture_data():
    def _load_fixture_data(name):
        import numpy as np

        path = os.path.join(os.path.dirname(__file__), 'fixtures', name)
        return np.load(path)

    return _load_fixture_data


@pytest.fixture
def matplotlib_figure():
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use('Agg')
    plt.figure()

    yield

    plt.close()
