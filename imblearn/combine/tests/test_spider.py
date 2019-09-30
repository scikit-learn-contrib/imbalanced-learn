"""Test the module SPIDER."""
# Authors: Matthew Eding
# License: MIT

import pytest
import numpy as np
from scipy import sparse

from sklearn.utils.testing import assert_allclose, assert_array_equal

from imblearn.combine import SPIDER


X = np.array([
    [-11.83, -6.81],
    [-11.72, -2.34],
    [-11.43, -5.85],
    [-10.66, -4.33],
    [-9.64, -7.05],
    [-8.39, -4.41],
    [-8.07, -5.66],
    [-7.28, 0.91],
    [-7.24, -2.41],
    [-6.13, -4.81],
    [-5.92, -6.81],
    [-4., -1.81],
    [-3.96, 2.67],
    [-3.74, -7.31],
    [-2.96, 4.69],
    [-1.56, -2.33],
    [-1.02, -4.57],
    [0.46, 4.07],
    [1.2, -1.53],
    [1.32, 0.41],
    [1.56, -5.19],
    [2.52, 5.89],
    [3.03, -4.15],
    [4., -0.59],
    [4.4, 2.07],
    [4.41, -7.45],
    [4.45, -4.12],
    [5.13, -6.28],
    [5.4, -5],
    [6.26, 4.65],
    [7.02, -6.22],
    [7.5, -0.11],
    [8.1, -2.05],
    [8.42, 2.47],
    [9.62, 3.87],
    [10.54, -4.47],
    [11.42, 0.01]
])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0,
              0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0])


RND_SEED = 0
R_TOL = 1e-4

def test_weak():
    X_expected = np.array([
        [3.03, -4.15],
        [-3.96, 2.67],
        [-3.96, 2.67],
        [-3.96, 2.67],
        [-11.83, -6.81],
        [-11.72, -2.34],
        [-11.43, -5.85],
        [-10.66, -4.33],
        [-9.64, -7.05],
        [-8.39, -4.41],
        [-8.07, -5.66],
        [-7.28, 0.91],
        [-7.24, -2.41],
        [-6.13, -4.81],
        [-5.92, -6.81],
        [-4., -1.81],
        [-3.96, 2.67],
        [-3.74, -7.31],
        [-2.96, 4.69],
        [-1.56, -2.33],
        [-1.02, -4.57],
        [0.46, 4.07],
        [1.2, -1.53],
        [1.32, 0.41],
        [1.56, -5.19],
        [3.03, -4.15],
        [4., -0.59],
        [4.4, 2.07],
        [4.41, -7.45],
        [5.13, -6.28],
        [5.4, -5.],
        [6.26, 4.65],
        [7.02, -6.22],
        [8.1, -2.05],
        [8.42, 2.47],
        [10.54, -4.47],
        [11.42, 0.01]
    ])
    y_expected = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                           0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                           1, 0, 0])

    weak = SPIDER(kind='weak')
    X_resampled, y_resampled = weak.fit_resample(X, y)

    assert_allclose(X_resampled, X_expected, rtol=R_TOL)
    assert_array_equal(y_resampled, y_expected)


def test_relabel():
    X_expected = np.array([
        [3.03, -4.15],
        [-3.96, 2.67],
        [-3.96, 2.67],
        [-3.96, 2.67],
        [-11.83, -6.81],
        [-11.72, -2.34],
        [-11.43, -5.85],
        [-10.66, -4.33],
        [-9.64, -7.05],
        [-8.39, -4.41],
        [-8.07, -5.66],
        [-7.28, 0.91],
        [-7.24, -2.41],
        [-6.13, -4.81],
        [-5.92, -6.81],
        [-4., -1.81],
        [-3.96, 2.67],
        [-3.74, -7.31],
        [-2.96, 4.69],
        [-1.56, -2.33],
        [-1.02, -4.57],
        [0.46, 4.07],
        [1.2, -1.53],
        [1.32, 0.41],
        [1.56, -5.19],
        [3.03, -4.15],
        [4., -0.59],
        [4.4, 2.07],
        [4.41, -7.45],
        [4.45, -4.12],
        [5.13, -6.28],
        [5.4, -5.],
        [6.26, 4.65],
        [7.02, -6.22],
        [7.5, -0.11],
        [8.1, -2.05],
        [8.42, 2.47],
        [9.62, 3.87],
        [10.54, -4.47],
        [11.42, 0.01]
    ])
    y_expected = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                           0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                           1, 0, 1, 1, 0, 0])

    relabel = SPIDER(kind='relabel')
    X_resampled, y_resampled = relabel.fit_resample(X, y)

    assert_allclose(X_resampled, X_expected, rtol=R_TOL)
    assert_array_equal(y_resampled, y_expected)


def test_strong():
    X_expected = np.array([
        [1.2, -1.53],
        [3.03, -4.15],
        [8.42, 2.47],
        [-3.96, 2.67],
        [-3.96, 2.67],
        [-3.96, 2.67],
        [-3.96, 2.67],
        [-3.96, 2.67],
        [-11.83, -6.81],
        [-11.72, -2.34],
        [-11.43, -5.85],
        [-10.66, -4.33],
        [-9.64, -7.05],
        [-8.39, -4.41],
        [-8.07, -5.66],
        [-7.28, 0.91],
        [-7.24, -2.41],
        [-6.13, -4.81],
        [-5.92, -6.81],
        [-4., -1.81],
        [-3.96, 2.67],
        [-3.74, -7.31],
        [-2.96, 4.69],
        [-1.56, -2.33],
        [-1.02, -4.57],
        [0.46, 4.07],
        [1.2, -1.53],
        [1.32, 0.41],
        [1.56, -5.19],
        [3.03, -4.15],
        [4., -0.59],
        [4.4, 2.07],
        [4.41, -7.45],
        [5.13, -6.28],
        [5.4, -5.],
        [6.26, 4.65],
        [7.02, -6.22],
        [8.1, -2.05],
        [8.42, 2.47],
        [10.54, -4.47],
        [11.42, 0.01]
    ])
    y_expected = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                           0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1,
                           1, 1, 1, 0, 1, 0, 0])

    strong = SPIDER(kind='strong')
    X_resampled, y_resampled = strong.fit_resample(X, y)

    assert_allclose(X_resampled, X_expected, rtol=R_TOL)
    assert_array_equal(y_resampled, y_expected)