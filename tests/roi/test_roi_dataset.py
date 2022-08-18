import pytest
import numpy as np

from gridfit.roi import ROIDataset, CircularROI, SquareROI
from gridfit.utils.image_moments import centroid, rms_size


def test_initialize_sets_data():
    data = np.arange(9).reshape(3, 3)
    rois = [SquareROI((1, 2), 3)]
    roi_dataset = ROIDataset(data, rois)

    assert np.all(roi_dataset.data == data)


def test_initialize_sets_rois():
    data = np.arange(9).reshape(3, 3)
    rois = [SquareROI((1, 2), 3)]
    roi_dataset = ROIDataset(data, rois)

    assert roi_dataset.rois == rois


def test_initialize_raises_error_when_data_is_not_numpy_array():
    with pytest.raises(ValueError):
        ROIDataset('test', [SquareROI((1, 2), 3)])


def test_initialize_raises_error_when_data_is_not_2d():
    with pytest.raises(ValueError):
        ROIDataset(np.arange(9), [SquareROI((1, 2), 3)])


def test_initialize_raises_error_when_rois_is_not_list():
    with pytest.raises(ValueError):
        ROIDataset(np.arange(9).reshape(3, 3), 'test')


def test_initialize_raises_error_when_rois_is_empty():
    with pytest.raises(ValueError):
        ROIDataset(np.arange(9).reshape(3, 3), [])


def test_initialize_raises_error_when_rois_is_not_list_of_rois():
    with pytest.raises(ValueError):
        ROIDataset(np.arange(9).reshape(3, 3), ['test'])


def test_data_returns_data():
    data = np.arange(9).reshape(3, 3)
    rois = [SquareROI((1, 2), 3)]
    roi_dataset = ROIDataset(data, rois)

    assert np.all(roi_dataset.data == data)


def test_rois_returns_rois():
    data = np.arange(9).reshape(3, 3)
    rois = (SquareROI((1, 2), 3),)
    roi_dataset = ROIDataset(data, rois)

    assert roi_dataset.rois == rois


def test_rois_returns_tuple():
    data = np.arange(100).reshape(10, 10)
    rois = [CircularROI((3, 4), 2), CircularROI((5, 5), 2)]
    roi_dataset = ROIDataset(data, rois)

    assert isinstance(roi_dataset.rois, tuple)


def test_to_array_returns_array_with_roi_data():
    data = np.arange(100).reshape(10, 10)
    rois = [SquareROI((3, 4), 3), SquareROI((5, 5), 3)]
    roi_dataset = ROIDataset(data, rois)

    expected_array = np.array([rois[0].apply(data), rois[1].apply(data)])

    assert np.all(roi_dataset.to_array() == expected_array)


def test_to_array_returns_array_with_masked_roi_data():
    data = np.arange(100).reshape(10, 10)
    rois = [CircularROI((3, 4), 2), CircularROI((5, 5), 2)]
    roi_dataset = ROIDataset(data, rois)

    expected_mask = ~np.array([rois[0].mask, rois[1].mask])

    assert np.all(roi_dataset.to_array().mask == expected_mask)


def test_apply_returns_result_of_function_applied_to_each_roi_data():
    data = np.arange(100).reshape(10, 10)
    rois = [CircularROI((3, 4), 2), CircularROI((5, 5), 2)]
    roi_dataset = ROIDataset(data, rois)

    def func(x):
        return x[2, 2]

    expected_result = np.array([r.apply(data)[2, 2] for r in rois])

    assert np.all(roi_dataset.apply(func, compress=False) == expected_result)


def test_sum_returns_sum_of_each_roi_data():
    data = np.arange(100).reshape(10, 10)
    rois = [CircularROI((3, 4), 2), CircularROI((5, 5), 2)]
    roi_dataset = ROIDataset(data, rois)

    expected_result = np.array([r.apply(data).sum() for r in rois])

    assert np.all(roi_dataset.sum() == expected_result)


def test_min_returns_min_of_each_roi_data():
    data = np.arange(100).reshape(10, 10)
    rois = [CircularROI((3, 4), 2), CircularROI((5, 5), 2)]
    roi_dataset = ROIDataset(data, rois)

    expected_result = np.array([r.apply(data).min() for r in rois])

    assert np.all(roi_dataset.min() == expected_result)


def test_max_returns_max_of_each_roi_data():
    data = np.arange(100).reshape(10, 10)
    rois = [CircularROI((3, 4), 2), CircularROI((5, 5), 2)]
    roi_dataset = ROIDataset(data, rois)

    expected_result = np.array([r.apply(data).max() for r in rois])

    assert np.all(roi_dataset.max() == expected_result)


def test_mean_returns_mean_of_each_roi_data():
    data = np.arange(100).reshape(10, 10)
    rois = [CircularROI((3, 4), 2), CircularROI((5, 5), 2)]
    roi_dataset = ROIDataset(data, rois)

    expected_result = np.array([r.apply(data).mean() for r in rois])

    assert np.all(roi_dataset.mean() == expected_result)


def test_var_returns_var_of_each_roi_data():
    data = np.arange(100).reshape(10, 10)
    rois = [CircularROI((3, 4), 2), CircularROI((5, 5), 2)]
    roi_dataset = ROIDataset(data, rois)

    expected_result = np.array([r.apply(data).var() for r in rois])

    assert np.all(roi_dataset.var() == expected_result)


def test_std_returns_std_of_each_roi_data():
    data = np.arange(100).reshape(10, 10)
    rois = [CircularROI((3, 4), 2), CircularROI((5, 5), 2)]
    roi_dataset = ROIDataset(data, rois)

    expected_result = np.array([r.apply(data).std() for r in rois])

    assert np.all(roi_dataset.std() == expected_result)


def test_centroid_returns_centroid_of_each_roi_data():
    data = np.random.rand(100).reshape(10, 10)
    rois = [CircularROI((3, 4), 2), CircularROI((5, 5), 2)]
    roi_dataset = ROIDataset(data, rois)

    expected_result = np.array(
        [centroid(r.apply(data).filled(0)) for r in rois])

    assert np.all(roi_dataset.centroid() == expected_result)


def test_centroid_returns_centroid_of_each_roi_data_with_absolute_coordinates():  # noqa: E501
    data = np.random.rand(100).reshape(10, 10)
    rois = [CircularROI((3, 4), 2), CircularROI((5, 5), 2)]
    roi_dataset = ROIDataset(data, rois)

    expected_result = np.array(
        [centroid(r.apply(data).filled(0)) + r.boundaries[0] for r in rois])

    assert np.all(roi_dataset.centroid(absolute=True) == expected_result)


def test_rms_size_returns_rms_size_of_each_roi_data():
    data = np.random.rand(100).reshape(10, 10)
    rois = [CircularROI((3, 4), 2), CircularROI((5, 5), 2)]
    roi_dataset = ROIDataset(data, rois)

    expected_result = np.array(
        [rms_size(r.apply(data).filled(0)) for r in rois])

    assert np.all(roi_dataset.rms_size() == expected_result)


def test_plot_accepts_ax(matplotlib_figure):
    import matplotlib.pyplot as plt

    data = np.random.rand(100).reshape(10, 10)
    rois = [CircularROI((3, 4), 2), CircularROI((5, 5), 2)]
    roi_dataset = ROIDataset(data, rois)

    ax = plt.gca()
    roi_dataset.plot(ax=ax)


def test_plot_accepts_imshow_kwargs(matplotlib_figure):
    data = np.random.rand(100).reshape(10, 10)
    rois = [CircularROI((3, 4), 2), CircularROI((5, 5), 2)]
    roi_dataset = ROIDataset(data, rois)

    roi_dataset.plot(imshow_kwargs=dict(cmap='jet'))


def test_plot_accepts_kwargs(matplotlib_figure):
    data = np.random.rand(100).reshape(10, 10)
    rois = [CircularROI((3, 4), 2), CircularROI((5, 5), 2)]
    roi_dataset = ROIDataset(data, rois)

    roi_dataset.plot(show_center=True)
