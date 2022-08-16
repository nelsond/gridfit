import numpy as np


def rotate_point(point, angle, center=np.zeros(2)):
    """
    Rotates a point in two dimensions around the given angle and center
    coordinates.

    Arguments:
        point (numpy.ndarray):
            Coordinates of the point as array of length two.

        angle (float):
            The rotation angle in degrees.

        center (numpy.ndarray, optional):
            The coordinates around which the point is rotated, origin (0, 0)
            by default.

    Returns:
        numpy.ndarray:
            The point rotated around the given angle and center coordinates.
    """
    if angle == 0:
        return point

    phi = angle / 180 * np.pi
    cos_phi, sin_phi = np.cos(phi), np.sin(phi)

    R = np.array([[cos_phi, -sin_phi],
                  [sin_phi, cos_phi]])

    return (R @ (point - center)) + center
