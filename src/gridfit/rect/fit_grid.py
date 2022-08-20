from typing import Any, Union, Tuple
import numpy as np
import numpy.typing as npt
from scipy import optimize, signal

from ..utils import cartesian_product, find_center, rotate, rotate_point
from ..funcs import gaussians_n


def fit_peaks(
    data: npt.NDArray[np.float_],
    axis: int = 0,
    min_rel_height: float = 0.25,
    min_rel_distance: float = 0.01
) -> npt.NDArray[np.float_]:
    """
    Fit peaks along a single axis in a two-dimensional array using integration
    and the sum of equidistant Gaussians.

    Arguments:
        data (numpy.ndarray):
            The data to fit.

        axis (int, optional):
            Axis along which to integrate before fitting, 0 by default.

        min_rel_height (float, optional):
            Minimum height of the peaks to the maximum height of the data,
            0.25 by default.

        min_rel_distance (float, optional):
            Minimum relative distance between the peaks, 0.01 by default.

    Returns:
        numpy.ndarray:
            The coordinates of the fitted peaks.
    """
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


def fit_grid(
    data: npt.NDArray[np.float_],
    angle: float = 0,
    full_output: bool = False,
    debug: bool = False,
    **kwargs: Any
) -> Union[
    npt.NDArray[np.float_],
    Tuple[
        npt.NDArray[np.float_],
        npt.NDArray[np.float_],
        npt.NDArray[np.float_]
    ]
]:
    """
    Fit a rectangular grid to the data of a two-dimensional array.

    Arguments:
        data (numpy.ndarray):
            The data to fit.

        angle (float, optional):
            The angle of the grid in degrees, 0 by default.

        full_output (bool, optional):
            Whether to return the x and y coordinates in addition to the grid,
            False by default.

        debug (bool, optional):
            Whether to show plot of data and the fitted grid, False by default.
            Note that this requires matplotlib.

        **kwargs:
            Keyword arguments are passed to gridfit.rect.fit_peaks.

    Returns:
        numpy.ndarray:
            Rotated grid with shape (n, m, 2) where n and m corresponds to the
            number of peaks along the first and second axis, respectively.

        numpy.ndarray, numpy.ndarray, numpy.ndarray (full_output set to True):
            The first two numpy arrays contain the x and y coordinates of the
            grid without the rotation applied. The last numpy array contains
            the rotated grid (see above for details).
    """
    data_rotated = rotate(data, angle)

    x = fit_peaks(data_rotated, axis=1, **kwargs)
    y = fit_peaks(data_rotated, axis=0, **kwargs)

    prod = cartesian_product(np.array(x), np.array(y))

    if angle != 0:
        center = find_center(data)

        for i, col in enumerate(prod):
            for j, row in enumerate(col):
                prod[i][j] = rotate_point(row, -angle, center)

    if debug:
        import matplotlib.pyplot as plt

        for point in prod.reshape(-1, 2):
            plt.plot(*point[::-1], 'ro', mfc='none', ms=6, mew=1)

        plt.imshow(data)

    if full_output:
        return x, y, prod

    return prod
