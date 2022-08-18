# gridfit [![codecov](https://codecov.io/gh/nelsond/gridfit/branch/main/graph/badge.svg?token=BCA7549I1W)](https://codecov.io/gh/nelsond/gridfit)

Simple Python library for fitting a (rectangular) grid of shapes in an image to a set of points.

**Note that the current beta version still lacks a number of features.**

## Requirements

This module requires Python >= 3.6 and the following packages:

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

After finding the dominant angle of the rectangular grid, we can fit Gaussian-shaped blobs to the data and extract the coordinates of all points in the grid.

```python
import matplotlib.pyplot as plt
from gridfit.rect import fit_grid

grid = fit_grid(image, angle)

plt.imshow(image)
for i, col in enumerate(grid):
    for j, row in enumerate(col):
        plt.plot(row[0], row[1], 'ro', mfc='none', ms=10)
```

## Development

Install requirements for development environment

```shell
$ pip install -r requirements/dev.txt
```

Run tests

```shell
$ pytest tests/
```

Generate coverage report

```shell
$ pytest --cov=gridfit --cov-report html tests/
```