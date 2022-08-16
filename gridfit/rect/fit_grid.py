import numpy as np
from scipy import optimize, signal

from ..utils import cartesian_product, find_center, rotate, rotate_point
from ..funcs import gaussians_n


def fit_peaks(data, axis=0, min_rel_height=0.25, min_rel_distance=0.01):
    integrated = data.sum(axis=axis)
    point_count = len(integrated)

    peaks, params = signal.find_peaks(
        integrated, height=integrated.max() * min_rel_height,
        distance=max(int(point_count * min_rel_distance), 1))
    peak_heights = params['peak_heights']

    x = np.arange(len(integrated))
    dx = np.diff(peaks).mean()
    guess = (peaks[0], dx, dx / 10, 0, *peak_heights)
    popt, _ = optimize.curve_fit(gaussians_n, x, integrated, p0=guess)
    x_0, dx = popt[:2]

    return x_0 + dx * np.arange(0, len(popt[4:]))


def fit_grid(data, angle=0, debug=False, full_output=False, **kwargs):
    if angle != 0:
        data_rotated = rotate(data, angle)
    else:
        data_rotated = data

    x = fit_peaks(data_rotated, axis=0, **kwargs)
    y = fit_peaks(data_rotated, axis=1, **kwargs)

    prod = cartesian_product(np.array(x), np.array(y))

    if angle != 0:
        center = find_center(data)

        for i, col in enumerate(prod):
            for j, row in enumerate(col):
                prod[i][j] = rotate_point(row, angle, center[::-1])

    if debug:
        import matplotlib.pyplot as plt

        for col in prod:
            for point in col:
                plt.plot(*point, 'ro', mfc='none', ms=6)

        plt.imshow(data)
        plt.tight_layout()

    if full_output:
        return x, y, prod

    return prod
