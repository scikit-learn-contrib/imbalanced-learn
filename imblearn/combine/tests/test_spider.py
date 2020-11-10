"""Test the module SPIDER."""
# Authors: Matthew Eding
# License: MIT

import pytest
import numpy as np

from sklearn.neighbors import NearestNeighbors
from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_array_equal

from imblearn.combine import SPIDER


RND_SEED = 0
X = np.array(
    [
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
    ]
)
y = np.array(
    [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0,
        0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0
    ]
)
R_TOL = 1e-4


def test_spider_init():
    spider = SPIDER()
    assert spider.n_neighbors == 3
    assert spider.additional_neighbors == 2
    assert spider.kind_sel == "weak"
    assert spider.n_jobs is None


def test_spider_weak():
    weak = SPIDER(kind_sel="weak")
    X_resampled, y_resampled = weak.fit_resample(X, y)
    X_gt = np.array(
        [
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
        ]
    )
    y_gt = np.array(
        [
            1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            1, 0, 0
        ]
    )
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_spider_relabel():
    relabel = SPIDER(kind_sel="relabel")
    X_resampled, y_resampled = relabel.fit_resample(X, y)
    X_gt = np.array(
        [
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
        ]
    )
    y_gt = np.array(
        [
            1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1,
            1, 0, 1, 1, 0, 0
        ]
    )
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_spider_strong():
    strong = SPIDER(kind_sel="strong")
    X_resampled, y_resampled = strong.fit_resample(X, y)
    X_gt = np.array(
        [
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
        ]
    )
    y_gt = np.array(
        [
            1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1,
            1, 1, 1, 0, 1, 0, 0
        ]
    )
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_spider_wrong_kind_sel():
    spider = SPIDER(kind_sel="rand")
    with pytest.raises(ValueError, match='The possible "kind" of algorithm'):
        spider.fit_resample(X, y)


def test_spider_fit_resample_with_nn_object():
    nn = NearestNeighbors(n_neighbors=4)
    spider = SPIDER(n_neighbors=nn)
    X_resampled, y_resampled = spider.fit_resample(X, y)
    X_gt = np.array(
        [
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
        ]
    )
    y_gt = np.array(
        [
            1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0,
            1, 0, 0
        ]
    )
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_spider_not_good_object():
    nn = "rand"
    spider = SPIDER(n_neighbors=nn)
    with pytest.raises(ValueError, match="has to be one of"):
        spider.fit_resample(X, y)


@pytest.mark.parametrize(
    "add_neigh, err_type, err_msg",
    [
        (0, ValueError, "additional_neighbors must be at least 1"),
        (0.0, TypeError, "additional_neighbors must be an integer"),
        (2.0, TypeError, "additional_neighbors must be an integer"),
        ("2", TypeError, "additional_neighbors must be an integer"),
        (2 + 0j, TypeError, "additional_neighbors must be an integer"),
    ],
)
def test_spider_invalid_additional_neighbors(add_neigh, err_type, err_msg):
    spider = SPIDER(additional_neighbors=add_neigh)
    with pytest.raises(err_type, match=err_msg):
        spider.fit_resample(X, y)
