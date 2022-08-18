from multiprocessing.sharedctypes import Value
import numpy as np

from .circular_roi import CircularROI
from .square_roi import SquareROI
from ..utils.image_moments import centroid, rms_size


class ROIDataset:
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
        self._rois = rois
        self._roi_data = None

    @property
    def data(self):
        return self._data

    @property
    def rois(self):
        return self._rois

    def to_array(self):
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
        def apply_func(data):
            if compress is True:
                return func(data.compressed())

            return func(data)

        return np.array([apply_func(d) for d in self.to_array()])

    def sum(self):
        return self.apply(np.sum)

    def min(self):
        return self.apply(np.min)

    def max(self):
        return self.apply(np.max)

    def mean(self):
        return self.apply(np.mean)

    def var(self):
        return self.apply(np.var)

    def std(self):
        return self.apply(np.std)

    def centroid(self, absolute=False):
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
        def apply_rms_size(data):
            if isinstance(data, np.ma.MaskedArray):
                data = data.filled(0)

            return rms_size(data)

        return self.apply(apply_rms_size, compress=False)

    def plot(self, ax=None, imshow_kwargs={}, **kwargs):
        import matplotlib.pyplot as plt

        if ax is None:
            ax = plt.gca()

        ax.imshow(self._data, **imshow_kwargs)

        for roi in self.rois:
            roi.plot(**kwargs)
