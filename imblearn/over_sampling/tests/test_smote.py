"""Test the module SMOTE."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import print_function

import numpy as np
from pytest import raises

from sklearn.utils.testing import assert_allclose, assert_array_equal
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE


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


def test_smote_wrong_kind():
    kind = 'rnd'
    smote = SMOTE(kind=kind, random_state=RND_SEED)
    with raises(ValueError, match="Unknown kind for SMOTE"):
        smote.fit_sample(X, Y)


def test_sample_regular():
    kind = 'regular'
    smote = SMOTE(random_state=RND_SEED, kind=kind)
    X_resampled, y_resampled = smote.fit_sample(X, Y)
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
                     [0.29307743, -0.14670439], [0.84976473, -0.15570176],
                     [0.61319159, -0.11571668], [0.66052536, -0.28246517]])
    y_gt = np.array([
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0
    ])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_regular_half():
    ratio = 0.8
    kind = 'regular'
    smote = SMOTE(ratio=ratio, random_state=RND_SEED, kind=kind)
    X_resampled, y_resampled = smote.fit_sample(X, Y)
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
                     [0.36784496, -0.1953161]])
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_borderline1():
    kind = 'borderline1'
    smote = SMOTE(random_state=RND_SEED, kind=kind)
    X_resampled, y_resampled = smote.fit_sample(X, Y)
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
                     [0.3765279, -0.2009615], [0.55276636, -0.10550373],
                     [0.45413452, -0.08883319], [1.21118683, -0.22817957]])
    y_gt = np.array([
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0
    ])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_borderline2():
    kind = 'borderline2'
    smote = SMOTE(random_state=RND_SEED, kind=kind)
    X_resampled, y_resampled = smote.fit_sample(X, Y)
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
                     [0.47436888, -0.2645749], [1.07844561, -0.19435291],
                     [0.33339622, 0.49870937]])
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_svm():
    kind = 'svm'
    smote = SMOTE(random_state=RND_SEED, kind=kind)
    X_resampled, y_resampled = smote.fit_sample(X, Y)
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
                     [0.47436888, -0.2645749], [1.07844561, -0.19435291],
                     [1.44015515, -1.30621303]])
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_fit_sample_nn_obj():
    kind = 'borderline1'
    nn_m = NearestNeighbors(n_neighbors=11)
    nn_k = NearestNeighbors(n_neighbors=6)
    smote = SMOTE(random_state=RND_SEED, kind=kind, k_neighbors=nn_k,
                  m_neighbors=nn_m)
    X_resampled, y_resampled = smote.fit_sample(X, Y)
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
                     [0.3765279, -0.2009615], [0.55276636, -0.10550373],
                     [0.45413452, -0.08883319], [1.21118683, -0.22817957]])
    y_gt = np.array([
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0
    ])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_regular_with_nn():
    kind = 'regular'
    nn_k = NearestNeighbors(n_neighbors=6)
    smote = SMOTE(random_state=RND_SEED, kind=kind, k_neighbors=nn_k)
    X_resampled, y_resampled = smote.fit_sample(X, Y)
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
                     [0.29307743, -0.14670439], [0.84976473, -0.15570176],
                     [0.61319159, -0.11571668], [0.66052536, -0.28246517]])
    y_gt = np.array([
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0
    ])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_wrong_nn():
    kind = 'borderline1'
    nn_m = 'rnd'
    nn_k = NearestNeighbors(n_neighbors=6)
    smote = SMOTE(
        random_state=RND_SEED, kind=kind, k_neighbors=nn_k, m_neighbors=nn_m)
    with raises(ValueError, match="has to be one of"):
        smote.fit_sample(X, Y)
    nn_k = 'rnd'
    nn_m = NearestNeighbors(n_neighbors=10)
    smote = SMOTE(
        random_state=RND_SEED, kind=kind, k_neighbors=nn_k, m_neighbors=nn_m)
    with raises(ValueError, match="has to be one of"):
        smote.fit_sample(X, Y)
    kind = 'regular'
    nn_k = 'rnd'
    smote = SMOTE(random_state=RND_SEED, kind=kind, k_neighbors=nn_k)
    with raises(ValueError, match="has to be one of"):
        smote.fit_sample(X, Y)


def test_sample_regular_with_nn_svm():
    kind = 'svm'
    nn_k = NearestNeighbors(n_neighbors=6)
    svm = SVC(random_state=RND_SEED)
    smote = SMOTE(
        random_state=RND_SEED, kind=kind, k_neighbors=nn_k, svm_estimator=svm)
    X_resampled, y_resampled = smote.fit_sample(X, Y)
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
                     [0.47436888, -0.2645749], [1.07844561, -0.19435291],
                     [1.44015515, -1.30621303]])
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_regular_wrong_svm():
    kind = 'svm'
    nn_k = NearestNeighbors(n_neighbors=6)
    svm = 'rnd'
    smote = SMOTE(
        random_state=RND_SEED, kind=kind, k_neighbors=nn_k, svm_estimator=svm)

    with raises(ValueError, match="has to be one of"):
        smote.fit_sample(X, Y)
