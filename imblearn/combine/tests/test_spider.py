"""Test the module SPIDER."""
# Author: Matthew Eding
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
    [ -9.64, -7.05],
    [ -8.39, -4.41],
    [ -8.07, -5.66],
    [ -7.28,  0.91],
    [ -7.24, -2.41],
    [ -6.13, -4.81],
    [ -5.92, -6.81],
    [ -4.  , -1.81],
    [ -3.96,  2.67],
    [ -3.74, -7.31],
    [ -2.96,  4.69],
    [ -1.56, -2.33],
    [ -1.02, -4.57],
    [  0.46,  4.07],
    [  1.2 , -1.53],
    [  1.32,  0.41],
    [  1.56, -5.19],
    [  2.52,  5.89],
    [  3.03, -4.15],
    [  4.  , -0.59],
    [  4.4 ,  2.07],
    [  4.41, -7.45],
    [  4.45, -4.12],
    [  5.13, -6.28],
    [  5.4 , -5   ],
    [  6.26,  4.65],
    [  7.02, -6.22],
    [  7.5 , -0.11],
    [  8.1 , -2.05],
    [  8.42,  2.47],
    [  9.62,  3.87],
    [ 10.54, -4.47],
    [ 11.42,  0.01]
])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0,
    0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0])


RND_SEED = 0
R_TOL = 1e-4


#FIXME: failing test
#TODO: also parametrize 'kind'
@pytest.mark.parametrize('fmt', ['lil', 'csr', 'csc'])
def test_dense_sparse(fmt):
    X_spr = sparse.random(100, 10, format=fmt, random_state=0)
    X_arr = X_spr.toarray()

    random_state = np.random.RandomState(0)
    y = random_state.choice([0, 1], size=len(X_arr), p=[0.8, 0.2])

    spider = SPIDER()
    X_resampled_spr, y_resampled_spr = spider.fit_resample(X_spr, y)
    X_resampled_spr = X_resampled_spr.toarray()
    sort_spr_idxs = np.argsort(X_resampled_spr[:, 0], axis=0)

    X_resampled_arr, y_resampled_arr = spider.fit_resample(X_arr, y)
    sort_arr_idxs = np.argsort(X_resampled_arr[:, 0], axis=0)

    # sparse implementation amplifies in different order than dense
    assert_allclose(
        X_resampled_spr[sort_spr_idxs],
        X_resampled_arr[sort_arr_idxs],
        rtol=R_TOL
    )
    assert_array_equal(
        y_resampled_spr[sort_spr_idxs],
        y_resampled_arr[sort_arr_idxs]
    )


def test_weak():
    X_expected = np.array([
        [ -3.96,   2.67],
        [ -3.96,   2.67],
        [ -3.96,   2.67],
        [  3.03,  -4.15],
        [-11.83,  -6.81],
        [-11.72,  -2.34],
        [-11.43,  -5.85],
        [-10.66,  -4.33],
        [ -9.64,  -7.05],
        [ -8.39,  -4.41],
        [ -8.07,  -5.66],
        [ -7.28,   0.91],
        [ -7.24,  -2.41],
        [ -6.13,  -4.81],
        [ -5.92,  -6.81],
        [ -4.  ,  -1.81],
        [ -3.96,   2.67],
        [ -3.74,  -7.31],
        [ -2.96,   4.69],
        [ -1.56,  -2.33],
        [ -1.02,  -4.57],
        [  0.46,   4.07],
        [  1.2 ,  -1.53],
        [  1.32,   0.41],
        [  1.56,  -5.19],
        [  3.03,  -4.15],
        [  4.  ,  -0.59],
        [  4.4 ,   2.07],
        [  4.41,  -7.45],
        [  5.13,  -6.28],
        [  5.4 ,  -5.  ],
        [  6.26,   4.65],
        [  7.02,  -6.22],
        [  8.1 ,  -2.05],
        [  8.42,   2.47],
        [ 10.54,  -4.47],
        [ 11.42,   0.01]
    ])
    y_expected = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0])

    weak = SPIDER(kind='weak')
    X_resampled, y_resampled = weak.fit_resample(X, y)

    assert_allclose(X_resampled, X_expected, rtol=R_TOL)
    assert_array_equal(y_resampled, y_expected)


def test_relabel():
    X_expected = np.array([
        [ -3.96,   2.67],
        [ -3.96,   2.67],
        [ -3.96,   2.67],
        [  3.03,  -4.15],
        [-11.83,  -6.81],
        [-11.72,  -2.34],
        [-11.43,  -5.85],
        [-10.66,  -4.33],
        [ -9.64,  -7.05],
        [ -8.39,  -4.41],
        [ -8.07,  -5.66],
        [ -7.28,   0.91],
        [ -7.24,  -2.41],
        [ -6.13,  -4.81],
        [ -5.92,  -6.81],
        [ -4.  ,  -1.81],
        [ -3.96,   2.67],
        [ -3.74,  -7.31],
        [ -2.96,   4.69],
        [ -1.56,  -2.33],
        [ -1.02,  -4.57],
        [  0.46,   4.07],
        [  1.2 ,  -1.53],
        [  1.32,   0.41],
        [  1.56,  -5.19],
        [  3.03,  -4.15],
        [  4.  ,  -0.59],
        [  4.4 ,   2.07],
        [  4.41,  -7.45],
        [  4.45,  -4.12],
        [  5.13,  -6.28],
        [  5.4 ,  -5.  ],
        [  6.26,   4.65],
        [  7.02,  -6.22],
        [  7.5 ,  -0.11],
        [  8.1 ,  -2.05],
        [  8.42,   2.47],
        [  9.62,   3.87],
        [ 10.54,  -4.47],
        [ 11.42,   0.01]
    ])
    y_expected = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0])

    relabel = SPIDER(kind='relabel')
    X_resampled, y_resampled = relabel.fit_resample(X, y)

    assert_allclose(X_resampled, X_expected, rtol=R_TOL)
    assert_array_equal(y_resampled, y_expected)


def test_strong():
    X_expected = np.array([
        [  1.2 ,  -1.53],
        [  3.03,  -4.15],
        [ -3.96,   2.67],
        [ -3.96,   2.67],
        [ -3.96,   2.67],
        [ -3.96,   2.67],
        [ -3.96,   2.67],
        [  8.42,   2.47],
        [-11.83,  -6.81],
        [-11.72,  -2.34],
        [-11.43,  -5.85],
        [-10.66,  -4.33],
        [ -9.64,  -7.05],
        [ -8.39,  -4.41],
        [ -8.07,  -5.66],
        [ -7.28,   0.91],
        [ -7.24,  -2.41],
        [ -6.13,  -4.81],
        [ -5.92,  -6.81],
        [ -4.  ,  -1.81],
        [ -3.96,   2.67],
        [ -3.74,  -7.31],
        [ -2.96,   4.69],
        [ -1.56,  -2.33],
        [ -1.02,  -4.57],
        [  0.46,   4.07],
        [  1.2 ,  -1.53],
        [  1.32,   0.41],
        [  1.56,  -5.19],
        [  3.03,  -4.15],
        [  4.  ,  -0.59],
        [  4.4 ,   2.07],
        [  4.41,  -7.45],
        [  5.13,  -6.28],
        [  5.4 ,  -5.  ],
        [  6.26,   4.65],
        [  7.02,  -6.22],
        [  8.1 ,  -2.05],
        [  8.42,   2.47],
        [ 10.54,  -4.47],
        [ 11.42,   0.01]
    ])
    y_expected = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0])

    strong = SPIDER(kind='strong')
    X_resampled, y_resampled = strong.fit_resample(X, y)

    assert_allclose(X_resampled, X_expected, rtol=R_TOL)
    assert_array_equal(y_resampled, y_expected)
