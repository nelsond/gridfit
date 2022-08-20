from typing import Tuple, Union
import numpy as np
import numpy.typing as npt

from ..utils import rotate, auto_pad


def discrete_radon_transform(
    data: npt.NDArray[np.float_],
    steps: int = 100,
    axis: int = 0,
    angular_range: Union[float, Tuple[float, float]] = (-45., 45.),
    preprocess: bool = True
) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """
    Compute the discrete radon transform for a two-dimensional array.

    Arguments:
        data (numpy.ndarray):
            The image data.

        steps (int, optional):
            The number of angular steps to use, 100 by default.

        axis (int, optional):
            The axis along which to compute the radon transform, 0 by default.

        angular_range (tupleor float):
            The angular range to use (in degrees), (-45, 45) by default.

        preprocess (bool, optional):
            Whether to pad the data to ensure rotation does not crop the image,
            True by default.

    Returns:
        numpy.ndarray, numpy.ndarray:
            The angular steps and the radon transform.
    """
    if preprocess is True:
        data_padded = auto_pad(data).astype(float)
    else:
        data_padded = data

    if isinstance(angular_range, float) or isinstance(angular_range, int):
        data_rot = rotate(data_padded, angular_range)
        return np.array([angular_range]), np.sum(data_rot, axis=axis)

    angles = np.linspace(*angular_range, steps)
    n = data_padded.shape[0 if axis == 1 else 1]
    radon_data = np.zeros((steps, n), dtype=float)

    for i, angle in enumerate(angles):
        angle = angles[i]
        data_rot = rotate(data_padded, angle)
        radon_data[i, :] = np.sum(data_rot, axis=axis)

    return angles, radon_data
