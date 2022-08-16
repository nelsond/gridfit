import numpy as np


def gaussians_n(x, x_0, dx, sigma, bg, *amplitudes):
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
    summed = 0

    for i, amplitude in enumerate(amplitudes):
        summed += amplitude * np.exp(-(x - (x_0 + i * dx))**2 / (2 * sigma**2))

    return summed + bg
