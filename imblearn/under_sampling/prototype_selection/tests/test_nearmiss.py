"""Test the module nearmiss."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import print_function

import numpy as np
from pytest import raises

from sklearn.utils.testing import assert_array_equal
from sklearn.neighbors import NearestNeighbors

from imblearn.under_sampling import NearMiss

from imblearn.utils.testing import warns

RND_SEED = 0
X = np.array([[1.17737838, -0.2002118],
              [0.4960075, 0.86130762],
              [-0.05903827, 0.10947647],
              [0.91464286, 1.61369212],
              [-0.54619583, 1.73009918],
              [-0.60413357, 0.24628718],
              [0.45713638, 1.31069295],
              [-0.04032409, 3.01186964],
              [0.03142011, 0.12323596],
              [0.50701028, -0.17636928],
              [-0.80809175, -1.09917302],
              [-0.20497017, -0.26630228],
              [0.99272351, -0.11631728],
              [-1.95581933, 0.69609604],
              [1.15157493, -1.2981518]])
Y = np.array([1, 2, 1, 0, 2, 1, 2, 2, 1, 2, 0, 0, 2, 1, 2])

VERSION_NEARMISS = (1, 2, 3)


# FIXME remove at the end of the deprecation 0.4
def test_nearmiss_deprecation():
    nm = NearMiss(ver3_samp_ngh=3, version=3)
    with warns(DeprecationWarning, match="deprecated from 0.2"):
        nm.fit_sample(X, Y)


def test_nearmiss_wrong_version():
    version = 1000
    nm = NearMiss(version=version, random_state=RND_SEED)
    with raises(ValueError, match="must be 1, 2 or 3"):
        nm.fit_sample(X, Y)


def test_nm_wrong_nn_obj():
    ratio = 'auto'
    nn = 'rnd'
    nm = NearMiss(ratio=ratio, random_state=RND_SEED,
                  version=VERSION_NEARMISS,
                  return_indices=True,
                  n_neighbors=nn)
    with raises(ValueError, match="has to be one of"):
        nm.fit_sample(X, Y)
    nn3 = 'rnd'
    nn = NearestNeighbors(n_neighbors=3)
    nm3 = NearMiss(ratio=ratio, random_state=RND_SEED,
                   version=3, return_indices=True,
                   n_neighbors=nn, n_neighbors_ver3=nn3)
    with raises(ValueError, match="has to be one of"):
        nm3.fit_sample(X, Y)


def test_nm_fit_sample_auto():
    ratio = 'auto'
    X_gt = [np.array([[0.91464286, 1.61369212],
                      [-0.80809175, -1.09917302],
                      [-0.20497017, -0.26630228],
                      [-0.05903827, 0.10947647],
                      [0.03142011, 0.12323596],
                      [-0.60413357, 0.24628718],
                      [0.50701028, -0.17636928],
                      [0.4960075, 0.86130762],
                      [0.45713638, 1.31069295]]),
            np.array([[0.91464286, 1.61369212],
                      [-0.80809175, -1.09917302],
                      [-0.20497017, -0.26630228],
                      [-0.05903827, 0.10947647],
                      [0.03142011, 0.12323596],
                      [-0.60413357, 0.24628718],
                      [0.50701028, -0.17636928],
                      [0.4960075, 0.86130762],
                      [0.45713638, 1.31069295]]),
            np.array([[0.91464286, 1.61369212],
                      [-0.80809175, -1.09917302],
                      [-0.20497017, -0.26630228],
                      [1.17737838, -0.2002118],
                      [-0.60413357, 0.24628718],
                      [0.03142011, 0.12323596],
                      [1.15157493, -1.2981518],
                      [-0.54619583, 1.73009918],
                      [0.99272351, -0.11631728]])]
    y_gt = [np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])]
    for version_idx, version in enumerate(VERSION_NEARMISS):
        nm = NearMiss(ratio=ratio, random_state=RND_SEED,
                      version=version)
        X_resampled, y_resampled = nm.fit_sample(X, Y)
        assert_array_equal(X_resampled, X_gt[version_idx])
        assert_array_equal(y_resampled, y_gt[version_idx])


def test_nm_fit_sample_auto_indices():
    ratio = 'auto'
    X_gt = [np.array([[0.91464286, 1.61369212],
                      [-0.80809175, -1.09917302],
                      [-0.20497017, -0.26630228],
                      [-0.05903827, 0.10947647],
                      [0.03142011, 0.12323596],
                      [-0.60413357, 0.24628718],
                      [0.50701028, -0.17636928],
                      [0.4960075, 0.86130762],
                      [0.45713638, 1.31069295]]),
            np.array([[0.91464286, 1.61369212],
                      [-0.80809175, -1.09917302],
                      [-0.20497017, -0.26630228],
                      [-0.05903827, 0.10947647],
                      [0.03142011, 0.12323596],
                      [-0.60413357, 0.24628718],
                      [0.50701028, -0.17636928],
                      [0.4960075, 0.86130762],
                      [0.45713638, 1.31069295]]),
            np.array([[0.91464286, 1.61369212],
                      [-0.80809175, -1.09917302],
                      [-0.20497017, -0.26630228],
                      [1.17737838, -0.2002118],
                      [-0.60413357, 0.24628718],
                      [0.03142011, 0.12323596],
                      [1.15157493, -1.2981518],
                      [-0.54619583, 1.73009918],
                      [0.99272351, -0.11631728]])]
    y_gt = [np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])]
    idx_gt = [np.array([3, 10, 11, 2, 8, 5, 9, 1, 6]),
              np.array([3, 10, 11, 2, 8, 5, 9, 1, 6]),
              np.array([3, 10, 11, 0, 5, 8, 14, 4, 12])]
    for version_idx, version in enumerate(VERSION_NEARMISS):
        nm = NearMiss(ratio=ratio, random_state=RND_SEED,
                      version=version, return_indices=True)
        X_resampled, y_resampled, idx_under = nm.fit_sample(X, Y)
        assert_array_equal(X_resampled, X_gt[version_idx])
        assert_array_equal(y_resampled, y_gt[version_idx])
        assert_array_equal(idx_under, idx_gt[version_idx])


def test_nm_fit_sample_float_ratio():
    ratio = .7
    X_gt = [np.array([[0.91464286, 1.61369212],
                      [-0.80809175, -1.09917302],
                      [-0.20497017, -0.26630228],
                      [-0.05903827, 0.10947647],
                      [0.03142011, 0.12323596],
                      [-0.60413357, 0.24628718],
                      [1.17737838, -0.2002118],
                      [0.50701028, -0.17636928],
                      [0.4960075, 0.86130762],
                      [0.45713638, 1.31069295],
                      [0.99272351, -0.11631728]]),
            np.array([[0.91464286, 1.61369212],
                      [-0.80809175, -1.09917302],
                      [-0.20497017, -0.26630228],
                      [-0.05903827, 0.10947647],
                      [0.03142011, 0.12323596],
                      [-0.60413357, 0.24628718],
                      [1.17737838, -0.2002118],
                      [0.50701028, -0.17636928],
                      [0.4960075, 0.86130762],
                      [0.45713638, 1.31069295],
                      [0.99272351, -0.11631728]]),
            np.array([[0.91464286, 1.61369212],
                      [-0.80809175, -1.09917302],
                      [-0.20497017, -0.26630228],
                      [1.17737838, -0.2002118],
                      [-0.60413357, 0.24628718],
                      [0.03142011, 0.12323596],
                      [-0.05903827, 0.10947647],
                      [1.15157493, -1.2981518],
                      [-0.54619583, 1.73009918],
                      [0.99272351, -0.11631728],
                      [0.45713638, 1.31069295]])]
    y_gt = [np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
            np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]),
            np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])]

    for version_idx, version in enumerate(VERSION_NEARMISS):
        nm = NearMiss(ratio=ratio, random_state=RND_SEED,
                      version=version)
        X_resampled, y_resampled = nm.fit_sample(X, Y)
        assert_array_equal(X_resampled, X_gt[version_idx])
        assert_array_equal(y_resampled, y_gt[version_idx])


def test_nm_fit_sample_nn_obj():
    ratio = 'auto'
    nn = NearestNeighbors(n_neighbors=3)
    X_gt = [np.array([[0.91464286, 1.61369212],
                      [-0.80809175, -1.09917302],
                      [-0.20497017, -0.26630228],
                      [-0.05903827, 0.10947647],
                      [0.03142011, 0.12323596],
                      [-0.60413357, 0.24628718],
                      [0.50701028, -0.17636928],
                      [0.4960075, 0.86130762],
                      [0.45713638, 1.31069295]]),
            np.array([[0.91464286, 1.61369212],
                      [-0.80809175, -1.09917302],
                      [-0.20497017, -0.26630228],
                      [-0.05903827, 0.10947647],
                      [0.03142011, 0.12323596],
                      [-0.60413357, 0.24628718],
                      [0.50701028, -0.17636928],
                      [0.4960075, 0.86130762],
                      [0.45713638, 1.31069295]]),
            np.array([[0.91464286, 1.61369212],
                      [-0.80809175, -1.09917302],
                      [-0.20497017, -0.26630228],
                      [1.17737838, -0.2002118],
                      [-0.60413357, 0.24628718],
                      [0.03142011, 0.12323596],
                      [1.15157493, -1.2981518],
                      [-0.54619583, 1.73009918],
                      [0.99272351, -0.11631728]])]
    y_gt = [np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2]),
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])]
    for version_idx, version in enumerate(VERSION_NEARMISS):
        nm = NearMiss(ratio=ratio, random_state=RND_SEED,
                      version=version, n_neighbors=nn)
        X_resampled, y_resampled = nm.fit_sample(X, Y)
        assert_array_equal(X_resampled, X_gt[version_idx])
        assert_array_equal(y_resampled, y_gt[version_idx])
