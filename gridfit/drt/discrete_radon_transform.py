import numpy as np

from ..utils import rotate, auto_pad


def discrete_radon_transform(data, steps=100, axis=0, angular_range=(-45, 45),
                             preprocess=True):
    """
    Compute the discrete radon transform for a two-dimensional array.

    Arguments:
        data (numpy.ndarray):
            The image data.

        steps (int, optional):
            The number of angular steps to use, 100 by default.

        axis (int, optional):
            The axis along which to compute the radon transform, 0 by default.

        angular_range (tuple, optional):
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
        return angular_range, np.sum(data_rot, axis=axis)

    angles = np.linspace(*angular_range, steps)
    n = data_padded.shape[0 if axis == 1 else 1]
    radon_data = np.zeros((steps, n), dtype=float)

    for i, angle in enumerate(angles):
        angle = angles[i]
        data_rot = rotate(data_padded, angle)
        radon_data[i, :] = np.sum(data_rot, axis=axis)

    return angles, radon_data
