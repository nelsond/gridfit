from typing import Union
import numpy as np
import numpy.typing as npt


def gaussians_n(
    x: Union[float, npt.NDArray[np.float_]],
    x_0: float,
    dx: float,
    sigma: float,
    bg: float,
    *amplitudes: float
) -> Union[float, npt.NDArray[np.float_]]:
    """
    Calculate the sum of n Gaussians with an equidistant spacing and the same
    standard deviation but a different amplitude.

    Arguments:
        x (float or numpy.ndarray):
            The x values.

        x_0 (float):
            The center of the first Gaussian.

        dx (float):
            The distance between the centers of two consecutive Gaussians.

        sigma (float):
            The standard deviation of the Gaussians.

        bg (float):
            The constant background.

        *amplitudes (float):
            The amplitudes of the Gaussians.

    Returns:
        float or numpy.ndarray:
            The sum of the n Gaussians.
    """
    summed = np.zeros_like(x, dtype=np.float_)

    for i, amplitude in enumerate(amplitudes):
        summed += amplitude * np.exp(-(x - (x_0 + i * dx))**2 / (2 * sigma**2))

    return summed + bg
