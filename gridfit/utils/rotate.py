import cv2


def rotate(data, angle):
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

    h, w = data.shape
    R = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)

    return cv2.warpAffine(data, R, (w, h))
