"""Test the module SMOTE ENN."""
from __future__ import print_function

import numpy as np
from numpy.testing import (assert_allclose, assert_array_equal,
                           assert_raises_regex)

from imblearn.combine import SMOTEENN
from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.over_sampling import SMOTE

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


def test_sample_regular():
    # Create the object
    smote = SMOTEENN(random_state=RND_SEED)
    # Fit the data
    smote.fit(X, Y)

    X_resampled, y_resampled = smote.fit_sample(X, Y)

    X_gt = np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                     [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                     [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                     [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                     [0.29307743, -0.14670439], [0.84976473, -0.15570176],
                     [0.61319159, -0.11571668], [0.66052536, -0.28246517],
                     [-0.28162401, -2.10400981], [0.83680821, 1.72827342],
                     [0.08711622, 0.93259929]])
    y_gt = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_regular_half():
    # Create the object
    ratio = 0.8
    smote = SMOTEENN(ratio=ratio, random_state=RND_SEED)
    # Fit the data
    smote.fit(X, Y)

    X_resampled, y_resampled = smote.fit_sample(X, Y)

    X_gt = np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                     [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                     [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                     [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                     [0.36784496, -0.1953161], [-0.28162401, -2.10400981],
                     [0.83680821, 1.72827342], [0.08711622, 0.93259929]])
    y_gt = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    assert_allclose(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_validate_estimator_init():
    # Create a SMOTE and Tomek object
    smote = SMOTE(random_state=RND_SEED)
    enn = EditedNearestNeighbours(random_state=RND_SEED)

    smt = SMOTEENN(smote=smote, enn=enn, random_state=RND_SEED)

    X_resampled, y_resampled = smt.fit_sample(X, Y)

    X_gt = np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                     [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                     [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                     [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                     [0.29307743, -0.14670439], [0.84976473, -0.15570176],
                     [0.61319159, -0.11571668], [0.66052536, -0.28246517],
                     [-0.28162401, -2.10400981], [0.83680821, 1.72827342],
                     [0.08711622, 0.93259929]])
    y_gt = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_validate_estimator_default():
    smt = SMOTEENN(random_state=RND_SEED)

    X_resampled, y_resampled = smt.fit_sample(X, Y)

    X_gt = np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                     [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                     [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                     [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                     [0.29307743, -0.14670439], [0.84976473, -0.15570176],
                     [0.61319159, -0.11571668], [0.66052536, -0.28246517],
                     [-0.28162401, -2.10400981], [0.83680821, 1.72827342],
                     [0.08711622, 0.93259929]])
    y_gt = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_validate_estimator_deprecation():
    X_gt = np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                     [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                     [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                     [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                     [0.29307743, -0.14670439], [0.84976473, -0.15570176],
                     [0.61319159, -0.11571668], [0.66052536, -0.28246517],
                     [-0.28162401, -2.10400981], [0.83680821, 1.72827342],
                     [0.08711622, 0.93259929]])
    y_gt = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1])

    smt = SMOTEENN(random_state=RND_SEED, n_jobs=-1)
    X_resampled, y_resampled = smt.fit_sample(X, Y)
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)

    smt = SMOTEENN(random_state=RND_SEED, k=5)
    X_resampled, y_resampled = smt.fit_sample(X, Y)
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_error_wrong_object():
    # Create a SMOTE and Tomek object
    smote = 'rnd'
    enn = 'rnd'

    smt = SMOTEENN(smote=smote, random_state=RND_SEED)
    assert_raises_regex(ValueError, "smote needs to be a SMOTE",
                        smt.fit, X, Y)
    smt = SMOTEENN(enn=enn, random_state=RND_SEED)
    assert_raises_regex(ValueError, "enn needs to be an ",
                        smt.fit, X, Y)
