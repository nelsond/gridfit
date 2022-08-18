from multiprocessing.sharedctypes import Value
import numpy as np

from .circular_roi import CircularROI
from .square_roi import SquareROI
from ..utils.image_moments import centroid, rms_size


class ROIDataset:
    """
    Utility class for working with a set of ROIs on image data.

    Arguments:
        data (numpy.ndarray):
            Image data.

        rois (tuple or list):
            List of ROIs, also see roi.CircularROI and roi.SquareROI.
    """
    def __init__(self, data, rois):
        if not isinstance(data, np.ndarray):
            raise ValueError('Invalid data, must be a numpy array.')

        if data.ndim != 2:
            raise ValueError('Invalid data shape, must be two-dimensional.')

        if not isinstance(rois, (tuple, list)):
            raise ValueError('Invalid rois, must be a tuple or list.')

        if len(rois) == 0:
            raise ValueError('Invalid rois, must contain at least one ROI.')

        for roi in rois:
            if not isinstance(roi, (CircularROI, SquareROI)):
                raise ValueError('Invalid ROI class {}, must be either '
                                 'SquareROI or CircularROI.'.format(type(roi)))

        self._data = data
        self._rois = tuple(rois)
        self._roi_data = None

    @property
    def data(self):
        """Data (numpy.ndarray)."""
        return self._data

    @property
    def rois(self):
        """ROIs (tuple)."""
        return self._rois

    def to_array(self):
        """
        Convert data in each ROI to single array.

        Returns:
            numpy.ndarray:
                Array of data in each ROI.
        """
        if self._roi_data is None:
            needs_mask = False
            rois_data = []

            for roi in self.rois:
                roi_data = roi.apply(self.data)
                rois_data.append(roi_data)

                if not needs_mask and isinstance(roi_data, np.ma.MaskedArray):
                    needs_mask = True

            if needs_mask:
                self._roi_data = np.ma.array(rois_data)
            else:
                self._roi_data = np.array(rois_data)

        return self._roi_data

    def apply(self, func, compress=True):
        """
        Apply function to each ROI.

        Arguments:
            func (callable):
                Function to apply to each ROI.

            compress (bool, optional):
                Whether to compress the individual ROI data (remove masked
                values) before applying function, True by default.

        Returns:
            numpy.ndarray:
                Array of results from applying function to each ROI.
        """
        def apply_func(data):
            if compress is True:
                return func(data.compressed())

            return func(data)

        return np.array([apply_func(d) for d in self.to_array()])

    def sum(self):
        """
        Sum data in each ROI.

        Returns:
            numpy.ndarray:
                Array of sums of data in each ROI.
        """
        return self.apply(np.sum)

    def min(self):
        """
        Minimum value in each ROI.

        Returns:
            numpy.ndarray:
                Array of minimum values in each ROI.
        """
        return self.apply(np.min)

    def max(self):
        """
        Maximum value in each ROI.

        Returns:
            numpy.ndarray:
                Array of maximum values in each ROI.
        """
        return self.apply(np.max)

    def mean(self):
        """
        Mean value in each ROI.

        Returns:
            numpy.ndarray:
                Array of mean values in each ROI.
        """
        return self.apply(np.mean)

    def var(self):
        """
        Variance in each ROI.

        Returns:
            numpy.ndarray:
                Array of variances in each ROI.
        """
        return self.apply(np.var)

    def std(self):
        """
        Standard deviation in each ROI.

        Returns:
            numpy.ndarray:
                Array of standard deviations in each ROI.
        """
        return self.apply(np.std)

    def centroid(self, absolute=False):
        """
        Centroid of each ROI (first moment).

        Arguments:
            absolute (bool, optional):
                Whether to return the absolute centroid position (in the
                original data coordinates), False by default.

        Returns:
            numpy.ndarray:
                Array of centroids of each ROI.
        """
        if absolute is True:
            return np.array([roi.centroid for roi in self.rois])

        def apply_centroid(data):
            if isinstance(data, np.ma.MaskedArray):
                data = data.filled(0)

            return centroid(data)

        c = self.apply(apply_centroid, compress=False)

        if absolute:
            for i, roi in enumerate(self.rois):
                c[i] += roi.boundaries[0]

        return c

    def rms_size(self):
        """
        Root-mean-squared size of each ROI (square root of the second moment).

        Returns:
            numpy.ndarray:
                Array of root-mean-squared sizes of each ROI.
        """
        def apply_rms_size(data):
            if isinstance(data, np.ma.MaskedArray):
                data = data.filled(0)

            return rms_size(data)

        return self.apply(apply_rms_size, compress=False)

    def plot(self, ax=None, imshow_kwargs={}, **kwargs):
        """
        Plot data and ROIs.

        Arguments:
            ax (matplotlib.axes.Axes, optional):
                Axes to plot on, if None, a new figure and axes will be
                created, None by default.

            imshow_kwargs (dict, optional):
                Keyword arguments to pass to matplotlib.pyplot.imshow, empty
                by default.

            **kwargs:
                Keyword arguments to pass to plot method of each ROI, also see
                roi.SquareROI.plot and roi.CircularROI.plot
        """
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        ax.imshow(self._data, **imshow_kwargs)

        for roi in self.rois:
            roi.plot(**kwargs)
