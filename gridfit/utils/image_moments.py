import cv2
import numpy as np


def centroid(data):
    """
    Calculate the centroid of a two-dimensional array.

    Arguments:
        data (numpy.ndarray):
            The image data.

    Returns:
        tuple:
            The centroid along the first and second axis.
    """
    mom = cv2.moments(data)
    m00 = mom['m00']
    return mom['m01'] / m00, mom['m10'] / m00


def rms_size(data):
    """
    Calculate the root mean square size of a two-dimensional array.

    Arguments:
        data (numpy.ndarray):
            The image data.

    Returns:
        tuple:
            The root mean square size along the first and second axis.
    """
    mom = cv2.moments(data)
    m00 = mom['m00']
    return np.sqrt(mom['mu02'] / m00), np.sqrt(mom['mu20'] / m00)
