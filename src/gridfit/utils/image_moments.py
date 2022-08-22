import scipy.ndimage
import scipy.stats
import numpy as np
import numpy.typing as npt
from typing import Tuple


def centroid(
    data: npt.NDArray[np.float_]
) -> Tuple[float, float]:
    """
    Calculate the centroid of a two-dimensional array.

    Arguments:
        data (numpy.ndarray):
            The image data.

    Returns:
        tuple:
            The centroid along the first and second axis.
    """

    return scipy.ndimage.center_of_mass(data)

def rms_size(
    data: npt.NDArray[np.float_]
) -> Tuple[float, float]:
    """
    Calculate the root mean square size of a two-dimensional array.

    Arguments:
        data (numpy.ndarray):
            The image data.

    Returns:
        tuple:
            The root mean square size along the first and second axis.
    """

    center = centroid(data)
    M00 = np.sum(data)

    # calculate matrix mu(p,q) of all image moments up to order 2, i.e. p,q < 3
    order = 2
    calc = data.astype(np.float_, copy=False)
    for dim, dim_length in enumerate(data.shape):
        delta = np.arange(dim_length, dtype=np.float_) - center[dim]
        powers_of_delta = (
            delta[:, np.newaxis] ** np.arange(order+1, dtype=np.float_)
        )
        calc = np.moveaxis(calc, dim, data.ndim-1)
        calc = np.dot(calc, powers_of_delta)
        calc = np.moveaxis(calc, -1, dim)

    # return square root of second central moments along both directions  
    return (np.sqrt(calc[2,0]/M00), np.sqrt(calc[0,2]/M00))
