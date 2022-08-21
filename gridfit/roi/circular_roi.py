from typing import Optional, Tuple, Union
import numpy as np
import numpy.typing as npt
from matplotlib import patches, axes

from .square_roi import SquareROI


class CircularROI(SquareROI):
    """
    Utility class for working with a square region of interest (ROI).

    Arguments:
        center (list-like):
            The center coordinates of the ROI.

        radius (int):
            The radius of the ROI.
    """
    def __init__(
        self,
        center: Tuple[Union[float, int], Union[float, int]],
        radius: int
    ):
        self._radius = np.round(radius).astype(int)
        super().__init__(center, 2 * radius + 1)

    @property
    def radius(self) -> int:
        """Radius (int)."""
        return int(self._radius)

    @property
    def mask(self) -> npt.NDArray[np.bool_]:
        """Circular mask (numpy.ndarray)."""
        xx = np.arange(self.size) - self.size // 2
        yx = np.meshgrid(xx, xx)
        r = np.sqrt(yx[0]**2 + yx[1]**2)

        return r <= self.radius

    def apply(
        self,
        data: npt.NDArray[np.float_],
    ) -> npt.NDArray[np.float_]:
        """
        Extract data within ROI from a two-dimensional array.

        Note:
            This function returns a masked view of the passed numpy array.

        Arguments:
            data (numpy.ndarray):
                Two-dimensional array.

        Raises:
            ValueError:
                - If data is not a two-dimensional array.
                - If data is not a numpy.ndarray.
                - If boundaries are outside of data.

        Returns:
            numpy.ma.MaskedArray:
                The extracted data with a circular mask applied.
        """
        data_roi = super().apply(data)
        return np.ma.masked_array(
            data_roi, mask=~self.mask, fill_value=0, hard_mask=True)

    def plot(
        self,
        ax: Optional[axes.Axes] = None,
        color: str = 'r',
        lw: int = 1,
        show_center: bool = True
    ) -> patches.Circle:
        """
        Plot the boundaries of the ROI as a circle.

        Arguments:
            ax (matplotlib.axes.Axes, optional):
                The axes to plot on. If not given, the current axes are used.

            color (str, optional):
                The color of the circle, 'r' by default.

            lw (int, optional):
                The line width of the circle, 1 by default.

            show_center (bool, optional):
                If True, the center of the ROI is plotted as a dot, True by
                default.

        Returns:
            matplotlib.patch.Circle:
                The circle patch.
        """
        import matplotlib.pyplot as plt
        from matplotlib import patches

        if ax is None:
            ax = plt.gca()

        cy, cx = self.center
        center = (cx, cy)

        circle = patches.Circle(
            center, self.radius,
            fc='none', lw=lw, color=color)
        ax.add_patch(circle)

        if show_center is True:
            ax.plot(*center, '.', ms=1, color=color)

        return circle
