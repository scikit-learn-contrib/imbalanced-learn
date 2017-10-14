"""Test the module under sampler."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import print_function

import numpy as np
from pytest import raises

from sklearn.utils.testing import assert_allclose, assert_array_equal
from sklearn.neighbors import NearestNeighbors

from imblearn.over_sampling import ADASYN


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
    ratio = 'auto'
    ada = ADASYN(ratio=ratio, random_state=RND_SEED)
    assert ada.random_state == RND_SEED


def test_ada_fit():
    ada = ADASYN(random_state=RND_SEED)
    ada.fit(X, Y)
    assert ada.ratio_ == {0: 4, 1: 0}


def test_ada_fit_sample():
    ada = ADASYN(random_state=RND_SEED)
    X_resampled, y_resampled = ada.fit_sample(X, Y)
    X_gt = np.array([[0.11622591, -0.0317206],
                     [0.77481731, 0.60935141],
                     [1.25192108, -0.22367336],
                     [0.53366841, -0.30312976],
                     [1.52091956, -0.49283504],
                     [-0.28162401, -2.10400981],
                     [0.83680821, 1.72827342],
                     [0.3084254, 0.33299982],
                     [0.70472253, -0.73309052],
                     [0.28893132, -0.38761769],
                     [1.15514042, 0.0129463],
                     [0.88407872, 0.35454207],
                     [1.31301027, -0.92648734],
                     [-1.11515198, -0.93689695],
                     [-0.18410027, -0.45194484],
                     [0.9281014, 0.53085498],
                     [-0.14374509, 0.27370049],
                     [-0.41635887, -0.38299653],
                     [0.08711622, 0.93259929],
                     [1.70580611, -0.11219234],
                     [0.94899098, -0.30508981],
                     [0.28204936, -0.13953426],
                     [1.58028868, -0.04089947],
                     [0.66117333, -0.28009063]])
    y_gt = np.array([
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0
    ])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_ada_fit_ratio_error():
    ratio = 0.8
    ada = ADASYN(ratio=ratio, random_state=RND_SEED)
    with raises(ValueError, match="No samples will be generated."):
        ada.fit_sample(X, Y)


def test_ada_fit_sample_nn_obj():
    nn = NearestNeighbors(n_neighbors=6)
    ada = ADASYN(random_state=RND_SEED, n_neighbors=nn)
    X_resampled, y_resampled = ada.fit_sample(X, Y)
    X_gt = np.array([[0.11622591, -0.0317206],
                     [0.77481731, 0.60935141],
                     [1.25192108, -0.22367336],
                     [0.53366841, -0.30312976],
                     [1.52091956, -0.49283504],
                     [-0.28162401, -2.10400981],
                     [0.83680821, 1.72827342],
                     [0.3084254, 0.33299982],
                     [0.70472253, -0.73309052],
                     [0.28893132, -0.38761769],
                     [1.15514042, 0.0129463],
                     [0.88407872, 0.35454207],
                     [1.31301027, -0.92648734],
                     [-1.11515198, -0.93689695],
                     [-0.18410027, -0.45194484],
                     [0.9281014, 0.53085498],
                     [-0.14374509, 0.27370049],
                     [-0.41635887, -0.38299653],
                     [0.08711622, 0.93259929],
                     [1.70580611, -0.11219234],
                     [0.94899098, -0.30508981],
                     [0.28204936, -0.13953426],
                     [1.58028868, -0.04089947],
                     [0.66117333, -0.28009063]])
    y_gt = np.array([
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0
    ])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_ada_wrong_nn_obj():
    nn = 'rnd'
    ada = ADASYN(random_state=RND_SEED, n_neighbors=nn)
    with raises(ValueError, match="has to be one of"):
        ada.fit_sample(X, Y)
