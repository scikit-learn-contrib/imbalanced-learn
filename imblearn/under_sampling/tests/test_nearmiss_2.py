"""Test the module nearmiss."""
from __future__ import print_function

import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_raises
from sklearn.neighbors import NearestNeighbors

from imblearn.under_sampling import NearMiss

# Generate a global dataset to use
RND_SEED = 0
X = np.array([[1.17737838, -0.2002118], [0.4960075, 0.86130762],
              [-0.05903827, 0.10947647], [0.91464286, 1.61369212],
              [-0.54619583, 1.73009918], [-0.60413357, 0.24628718],
              [0.45713638, 1.31069295], [-0.04032409, 3.01186964],
              [0.03142011, 0.12323596], [0.50701028, -0.17636928],
              [-0.80809175, -1.09917302], [-0.20497017, -0.26630228],
              [0.99272351, -0.11631728], [-1.95581933, 0.69609604],
              [1.15157493, -1.2981518]])
Y = np.array([1, 2, 1, 0, 2, 1, 2, 2, 1, 2, 0, 0, 2, 1, 2])
VERSION_NEARMISS = 2


def test_nearmiss_wrong_version():
    version = 1000
    nm2 = NearMiss(version=version, random_state=RND_SEED)
    assert_raises(ValueError, nm2.fit_sample, X, Y)


def test_nearmiss_init():
    # Define a ratio
    ratio = 1.
    nm2 = NearMiss(
        ratio=ratio, random_state=RND_SEED, version=VERSION_NEARMISS)

    assert_equal(nm2.version, VERSION_NEARMISS)
    assert_equal(nm2.n_neighbors, 3)
    assert_equal(nm2.ratio, ratio)
    assert_equal(nm2.random_state, RND_SEED)


def test_nm2_fit_sample_auto():
    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    nm2 = NearMiss(
        ratio=ratio, random_state=RND_SEED, version=VERSION_NEARMISS)

    # Fit and sample
    X_resampled, y_resampled = nm2.fit_sample(X, Y)

    X_gt = np.array([[0.91464286, 1.61369212], [-0.80809175, -1.09917302],
                     [-0.20497017, -0.26630228], [-0.05903827, 0.10947647],
                     [0.03142011, 0.12323596], [-0.60413357, 0.24628718],
                     [0.50701028, -0.17636928], [0.4960075, 0.86130762],
                     [0.45713638, 1.31069295]])
    y_gt = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_nm2_fit_sample_auto_indices():
    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    nm2 = NearMiss(
        ratio=ratio,
        random_state=RND_SEED,
        version=VERSION_NEARMISS,
        return_indices=True)

    # Fit and sample
    X_resampled, y_resampled, idx_under = nm2.fit_sample(X, Y)

    X_gt = np.array([[0.91464286, 1.61369212], [-0.80809175, -1.09917302],
                     [-0.20497017, -0.26630228], [-0.05903827, 0.10947647],
                     [0.03142011, 0.12323596], [-0.60413357, 0.24628718],
                     [0.50701028, -0.17636928], [0.4960075, 0.86130762],
                     [0.45713638, 1.31069295]])
    y_gt = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    idx_gt = np.array([3, 10, 11, 2, 8, 5, 9, 1, 6])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


def test_nm2_fit_sample_half():
    # Define the parameter for the under-sampling
    ratio = .7

    # Create the object
    nm2 = NearMiss(
        ratio=ratio, random_state=RND_SEED, version=VERSION_NEARMISS)

    # Fit and sample
    X_resampled, y_resampled = nm2.fit_sample(X, Y)

    X_gt = np.array([[0.91464286, 1.61369212], [-0.80809175, -1.09917302],
                     [-0.20497017, -0.26630228], [-0.05903827, 0.10947647],
                     [0.03142011, 0.12323596], [-0.60413357, 0.24628718],
                     [1.17737838, -0.2002118], [0.50701028, -0.17636928],
                     [0.4960075, 0.86130762], [0.45713638, 1.31069295],
                     [0.99272351, -0.11631728]])
    y_gt = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_nm2_fit_sample_nn_obj():
    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    nn = NearestNeighbors(n_neighbors=3)
    nm2 = NearMiss(
        ratio=ratio,
        random_state=RND_SEED,
        version=VERSION_NEARMISS,
        return_indices=True,
        n_neighbors=nn)

    # Fit and sample
    X_resampled, y_resampled, idx_under = nm2.fit_sample(X, Y)

    X_gt = np.array([[0.91464286, 1.61369212], [-0.80809175, -1.09917302],
                     [-0.20497017, -0.26630228], [-0.05903827, 0.10947647],
                     [0.03142011, 0.12323596], [-0.60413357, 0.24628718],
                     [0.50701028, -0.17636928], [0.4960075, 0.86130762],
                     [0.45713638, 1.31069295]])
    y_gt = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    idx_gt = np.array([3, 10, 11, 2, 8, 5, 9, 1, 6])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


def test_nm2__wrong_nn_obj():
    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    nn = 'rnd'
    nm2 = NearMiss(
        ratio=ratio,
        random_state=RND_SEED,
        version=VERSION_NEARMISS,
        return_indices=True,
        n_neighbors=nn)

    # Fit and sample
    assert_raises(ValueError, nm2.fit_sample, X, Y)
