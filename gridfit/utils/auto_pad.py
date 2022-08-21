from typing import Union
import numpy as np
import numpy.typing as npt


def auto_pad(
    data: npt.NDArray[np.float_],
    value: Union[float, int] = 0
) -> npt.NDArray[np.float_]:
    """
    Pads two-dimensionaly array to make its shape match a square, where the
    length of a side is determined by the diagonal of the passed array.

    Arguments:
        data (numpy.ndarray):
            The datay array.

        value (optional):
            The fill value for padding, 0 by default.

    Returns:
        numpy.ndarray:
            The padded array.
    """
    height, width = data.shape
    diag = np.sqrt(height**2 + width**2)

    pad_y = int(np.ceil((diag - height) / 2))
    pad_x = int(np.ceil((diag - width) / 2))
    pad = ((pad_y, pad_y), (pad_x, pad_x))

    return np.pad(data, pad, constant_values=value)
