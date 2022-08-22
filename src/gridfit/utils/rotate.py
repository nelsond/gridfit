import cv2
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
        passed array.

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

    dst = np.zeros_like(data)
    h, w = data.shape
    R = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    cv2.warpAffine(data, R, (w, h), dst=dst)

    return dst
