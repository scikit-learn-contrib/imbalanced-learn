"""Test the module under sampler."""
from __future__ import print_function

import numpy as np
from numpy.testing import (assert_allclose, assert_array_equal,
                           assert_equal, assert_raises_regex)
from sklearn.neighbors import NearestNeighbors

from imblearn.over_sampling import ADASYN

# Generate a global dataset to use
RND_SEED = 0
X = np.array([[0.11622591, -0.0317206], [0.77481731, 0.60935141],
              [1.25192108, -0.22367336], [0.53366841, -0.30312976],
              [1.52091956, -0.49283504], [-0.28162401, -2.10400981],
              [0.83680821, 1.72827342], [0.3084254, 0.33299982],
              [0.70472253, -0.73309052], [0.28893132, -0.38761769],
              [1.15514042, 0.0129463], [0.88407872, 0.35454207],
              [1.31301027, -0.92648734], [-1.11515198, -0.93689695],
              [-0.18410027, -0.45194484], [0.9281014, 0.53085498],
              [-0.14374509, 0.27370049], [-0.41635887, -0.38299653],
              [0.08711622, 0.93259929], [1.70580611, -0.11219234]])
Y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
R_TOL = 1e-4


def test_ada_init():
    # Define a ratio
    ratio = 'auto'
    ada = ADASYN(ratio=ratio, random_state=RND_SEED)

    assert_equal(ada.random_state, RND_SEED)


def test_ada_fit():
    # Create the object
    ada = ADASYN(random_state=RND_SEED)
    # Fit the data
    ada.fit(X, Y)

    # Check if the data information have been computed
    assert_equal(ada.min_c_, 0)
    assert_equal(ada.maj_c_, 1)
    assert_equal(ada.stats_c_[0], 8)
    assert_equal(ada.stats_c_[1], 12)


def test_ada_fit_sample():
    # Resample the data
    ada = ADASYN(random_state=RND_SEED)
    X_resampled, y_resampled = ada.fit_sample(X, Y)

    X_gt = np.array([[0.11622591, -0.0317206], [0.77481731, 0.60935141],
                     [1.25192108, -0.22367336], [0.53366841, -0.30312976],
                     [1.52091956, -0.49283504], [-0.28162401, -2.10400981],
                     [0.83680821, 1.72827342], [0.3084254, 0.33299982],
                     [0.70472253, -0.73309052], [0.28893132, -0.38761769],
                     [1.15514042, 0.0129463], [0.88407872, 0.35454207],
                     [1.31301027, -0.92648734], [-1.11515198, -0.93689695],
                     [-0.18410027, -0.45194484], [0.9281014, 0.53085498],
                     [-0.14374509, 0.27370049], [-0.41635887, -0.38299653],
                     [0.08711622, 0.93259929], [1.70580611, -0.11219234],
                     [-0.06182085, -0.28084828], [0.38614986, -0.35405599],
                     [0.39635544, 0.33629036], [-0.24027923, 0.04116021]])
    y_gt = np.array([
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0
    ])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_ada_fit_sample_half():
    # Resample the data
    ratio = 0.8
    ada = ADASYN(ratio=ratio, random_state=RND_SEED)
    X_resampled, y_resampled = ada.fit_sample(X, Y)

    X_gt = np.array([[0.11622591, -0.0317206], [0.77481731, 0.60935141],
                     [1.25192108, -0.22367336], [0.53366841, -0.30312976],
                     [1.52091956, -0.49283504], [-0.28162401, -2.10400981],
                     [0.83680821, 1.72827342], [0.3084254, 0.33299982],
                     [0.70472253, -0.73309052], [0.28893132, -0.38761769],
                     [1.15514042, 0.0129463], [0.88407872, 0.35454207],
                     [1.31301027, -0.92648734], [-1.11515198, -0.93689695],
                     [-0.18410027, -0.45194484], [0.9281014, 0.53085498],
                     [-0.14374509, 0.27370049], [-0.41635887, -0.38299653],
                     [0.08711622, 0.93259929], [1.70580611, -0.11219234]])
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_ada_fit_sample_nn_obj():
    # Resample the data
    nn = NearestNeighbors(n_neighbors=6)
    ada = ADASYN(random_state=RND_SEED, n_neighbors=nn)
    X_resampled, y_resampled = ada.fit_sample(X, Y)

    X_gt = np.array([[0.11622591, -0.0317206], [0.77481731, 0.60935141],
                     [1.25192108, -0.22367336], [0.53366841, -0.30312976],
                     [1.52091956, -0.49283504], [-0.28162401, -2.10400981],
                     [0.83680821, 1.72827342], [0.3084254, 0.33299982],
                     [0.70472253, -0.73309052], [0.28893132, -0.38761769],
                     [1.15514042, 0.0129463], [0.88407872, 0.35454207],
                     [1.31301027, -0.92648734], [-1.11515198, -0.93689695],
                     [-0.18410027, -0.45194484], [0.9281014, 0.53085498],
                     [-0.14374509, 0.27370049], [-0.41635887, -0.38299653],
                     [0.08711622, 0.93259929], [1.70580611, -0.11219234],
                     [-0.06182085, -0.28084828], [0.38614986, -0.35405599],
                     [0.39635544, 0.33629036], [-0.24027923, 0.04116021]])
    y_gt = np.array([
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0
    ])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_ada_wrong_nn_obj():
    # Resample the data
    nn = 'rnd'
    ada = ADASYN(random_state=RND_SEED, n_neighbors=nn)
    assert_raises_regex(ValueError, "has to be one of",
                        ada.fit_sample, X, Y)
