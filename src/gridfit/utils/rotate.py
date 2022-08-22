import scipy.ndimage 
import numpy as np
import numpy.typing as npt


def rotate(
    data: npt.NDArray[np.float_],
    angle: float
) -> npt.NDArray[np.float_]:
    """
    Rotate a two-dimensional array by an arbitrary angle.

    Note:
        Note that the rotated data is cropped to match the dimensions of the
        passed array. Cubic interpolation is used.

    Arguments:
        data (numpy.ndarray):
            The data array.

        angle (float):
            The rotation angle in degrees.

    Returns:
        numpy.ndarray:
            The rotated array data.
    """
    if (angle % 360) == 0:
        return data

    rotated = scipy.ndimage.rotate(data, angle, reshape=False, order=3)

    return rotated
