from typing import Optional, Tuple
import numpy as np
import numpy.typing as npt
from matplotlib import patches, axes


class SquareROI:
    """
    Utility class for working with a square region of interest (ROI).

    Arguments:
        center (tuple):
            The center coordinates of the ROI.

        size (int):
            The size of the ROI.
    """
    def __init__(
        self,
        center: Tuple[float, float],
        size: int
    ):
        if not isinstance(center, (list, tuple, np.ndarray)):
            raise ValueError('Invalid center, must be a list or tuple.')

        if len(center) != 2:
            raise ValueError('Invalid center, must have length of two.')

        if not isinstance(center[0], (int, float)):
            raise ValueError(
                'Invalid first center coordinate, must be a number.')

        if not isinstance(center[1], (int, float)):
            raise ValueError(
                'Invalid second center coordinate, must be a number.')

        if not isinstance(size, (int, float)):
            raise ValueError('Invalid size, must be a number.')

        if size <= 0:
            raise ValueError('Invalid size, must be greater than zero.')

        self.center = np.array(center)
        self.size = size

    @property
    def center_rounded(self) -> npt.NDArray[np.int_]:
        """Rounded center coordinates (numpy.ndarray)."""
        return np.round(self.center).astype(int)

    @property
    def size_rounded(self) -> int:
        """Rounded size (int)."""
        return np.round(self.size).astype(int)

    @property
    def boundaries(self) -> npt.NDArray[np.int_]:
        """Boundaries, bottom left and top right corner (numpy.ndarray)."""
        center_rounded = self.center_rounded
        size_rounded = self.size_rounded

        half_size = size_rounded // 2
        x_0, y_0 = center_rounded - half_size
        x_1, y_1 = center_rounded + half_size

        if size_rounded % 2 == 1:
            x_1 += 1
            y_1 += 1

        return np.array(((x_0, y_0), (x_1, y_1)))

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape (tuple)."""
        return self.size_rounded, self.size_rounded

    def apply(
        self,
        data: npt.NDArray[np.float_]
    ) -> npt.NDArray[np.float_]:
        """
        Extract data within ROI from a two-dimensional array.

        Note:
            This function returns a view of the passed numpy array.

        Arguments:
            data (numpy.ndarray):
                Two-dimensional array.

        Raises:
            ValueError:
                - If data is not a two-dimensional array.
                - If data is not a numpy.ndarray.
                - If boundaries are outside of data.

        Returns:
            numpy.ndarray:
                The extracted data.
        """
        if not isinstance(data, np.ndarray):
            raise ValueError('Invalid data, must be a numpy array.')

        if data.ndim != 2:
            raise ValueError('Invalid data, must be two-dimensional.')

        bottom_left, top_right = self.boundaries

        if np.any(bottom_left < 0):
            raise ValueError('ROI is outside of data boundaries (negative'
                             ' bottom left coordinates).')

        if np.any(top_right > np.array(data.shape)):
            raise ValueError(
                'ROI is outside of data boundaries [top right coordinate {} '
                'is too large for data shape {}].'.format(
                    tuple(top_right.tolist()), data.shape))

        return data[bottom_left[0]:top_right[0], bottom_left[1]:top_right[1]]

    def plot(
        self,
        ax: Optional[axes.Axes] = None,
        color: str = 'r',
        lw: int = 1,
        show_center: bool = True
    ) -> patches.Rectangle:
        """
        Plot the boundaries of the ROI as a rectangle.

        Arguments:
            ax (matplotlib.axes.Axes, optional):
                The axes to plot on. If not given, the current axes are used.

            color (str, optional):
                The color of the rectangle, 'r' by default.

            lw (int, optional):
                The line width of the rectangle, 1 by default.

            show_center (bool, optional):
                If True, the center of the ROI is plotted as a dot, True by
                default.

        Returns:
            matplotlib.patch.Rectangle:
                The rectangle patch.
        """
        import matplotlib.pyplot as plt
        from matplotlib import patches

        if ax is None:
            ax = plt.gca()

        p_0, _ = self.boundaries
        rect = patches.Rectangle(
            p_0[::-1], self.size, self.size,
            fc='none', lw=lw, color=color)
        ax.add_patch(rect)

        if show_center is True:
            cy, cx = self.center
            ax.plot(cx, cy, '.', ms=1, color=color)

        return rect
