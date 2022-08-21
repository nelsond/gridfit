from typing import Tuple
import numpy as np
import numpy.typing as npt
from scipy import optimize

from ..utils import auto_pad
from ..drt import discrete_radon_transform


def find_dominant_angle(
    data: npt.NDArray[np.float_],
    angular_range: Tuple[float, float] = (-45., 45.),
    debug: bool = False
) -> float:
    """
    Find the dominant angle of a two-dimensional array.

    Note:
        This function employes the discrete Radon transform to estimate the
        dominant angle of an image. Use variable angular ranges if you want to
        find multiple dominant angles.

    Arguments:
        data (numpy.ndarray):
            The image data.

        angular_range (tuple, optional):
            The angular range to consider for the dominant angle (in degrees),
            (-45, 45) by default.

        debug (bool, optional):
            Show debug plots, False by default. Note that this requires
            matplotlib.

    Returns:
        float:
            The dominant angle.
    """
    data = auto_pad(data).astype(float)
    steps = int(2 * (max(angular_range) - min(angular_range)))

    angles, rt_data = discrete_radon_transform(
            data, axis=0, steps=steps, angular_range=angular_range,
            preprocess=False)

    std = np.std(rt_data, axis=-1)
    idx_max = np.argmax(std)
    guess = angles[idx_max]

    def opt_func(
        angle: float
    ) -> float:
        _, rt_data = discrete_radon_transform(
            data, axis=0, angular_range=angle, preprocess=False)
        inv = float(np.std(rt_data))

        if inv == 0:
            return np.inf

        return 1 / inv

    x_opt = optimize.golden(
        opt_func, brack=(guess - 2, guess + 2))

    if debug:
        import matplotlib.pyplot as plt

        angles, rt_data = discrete_radon_transform(
            data, axis=0, steps=100, angular_range=(guess - 2, guess + 2),
            preprocess=False)

        plt.plot(angles, rt_data.std(-1), c='0.8')
        y_opt = discrete_radon_transform(
            data, axis=0, angular_range=x_opt, preprocess=False)[1].std()
        plt.plot(x_opt, y_opt, 'ro')
        plt.plot(guess, std[idx_max], 'kx')
        plt.margins(x=0)

        plt.xlabel(r'Angle $\theta$ (deg)')
        plt.ylabel(r'Std. dev. of the Radon transform $\sigma(\theta)$')

        plt.tight_layout()

    return float(x_opt)
