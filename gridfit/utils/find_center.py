import numpy as np
import numpy.typing as npt


def find_center(
    data: npt.NDArray[np.float_]
) -> npt.NDArray[np.float_]:
    """
    Returns the center coordinates of a two-dimensional array.

    Arguments:
        data (two-dimensional numpy.ndarray):
            The data array.

    Returns:
        numpy.ndarray: The center coordinates.
    """
    h, w = data.shape
    return np.array([h / 2, w / 2])
