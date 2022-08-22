# gridfit

![github action](https://github.com/nelsond/gridfit/actions/workflows/ci.yml/badge.svg) [![codecov](https://codecov.io/gh/nelsond/gridfit/branch/main/graph/badge.svg?token=BCA7549I1W)](https://codecov.io/gh/nelsond/gridfit)

Simple Python library for fitting a (rectangular) grid of shapes in an image to a set of points.

**Note that the current beta version still lacks a number of features.**

![gridfit example](docs/header.png)

## Requirements

This module requires Python >= 3.8 and the following packages:

- `numpy`
- `scipy`
- `opencv-python`
- `matplotlib`

## Installation

Install with pip

```shell
$ pip install git+https://github.com/nelsond/gridfit
```

## Example usage

First, find the dominant angle of a rectangular grid in an image (uses the standard deviation of the Radon transformation).

```python
import numpy as np
from gridfit.rect import find_dominant_angle

image = np.load('data.npy')
angle = find_dominant_angle(image)
```

After finding the dominant angle of the rectangular grid, we can fit Gaussian-shaped blobs to the one-dimensional projection of the data.
This allows us to extract the coordinates of all points on the rectangular grid.

```python
import matplotlib.pyplot as plt
from gridfit.rect import fit_grid

grid = fit_grid(image, angle, min_rel_height=0.2)

plt.imshow(image)
for col in grid:
    for row in col:
        plt.plot(row[1], row[0], 'ro', mfc='none', ms=10)
```

Once the grid has been determined, we can use a set of ROIs to analyze the data.

```python
from gridfit.roi import CircularROI, ROIDataset

rois = []
for point in grid.reshape(-1, 2):
    roi = CircularROI(point, 15)
    rois.append(roi)

ds = ROIDataset(image, rois)
ds.plot(show_center=False, imshow_kwargs=dict(cmap=plt.cm.Greys_r))

# determine sum across each ROI on the grid
summed = ds.sum().reshape(*grid.shape[:2], -1)

# determine centroid in each ROI (first moment)
centroids = ds.centroid(absolute=True).reshape(*grid.shape[:2], -1)

# determine rms size in each ROI (square root of the second moment)
rms_size = ds.rms_size().reshape(*grid.shape[:2], -1)
```

## Development

Install requirements for development environment

```shell
$ pip install .[dev]
```

Run tests

```shell
$ pytest tests/
```

Generate coverage report

```shell
$ pytest --cov=gridfit --cov-report html tests/
```