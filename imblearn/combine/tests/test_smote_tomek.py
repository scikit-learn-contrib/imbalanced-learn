"""Test the module SMOTE ENN."""
from __future__ import print_function

import numpy as np
from numpy.testing import (assert_allclose, assert_array_equal,
                           assert_raises_regex)

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks

# Generate a global dataset to use
RND_SEED = 0
X = np.array([[0.20622591, 0.0582794], [0.68481731, 0.51935141],
              [1.34192108, -0.13367336], [0.62366841, -0.21312976],
              [1.61091956, -0.40283504], [-0.37162401, -2.19400981],
              [0.74680821, 1.63827342], [0.2184254, 0.24299982],
              [0.61472253, -0.82309052], [0.19893132, -0.47761769],
              [1.06514042, -0.0770537], [0.97407872, 0.44454207],
              [1.40301027, -0.83648734], [-1.20515198, -1.02689695],
              [-0.27410027, -0.54194484], [0.8381014, 0.44085498],
              [-0.23374509, 0.18370049], [-0.32635887, -0.29299653],
              [-0.00288378, 0.84259929], [1.79580611, -0.02219234]])
Y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
R_TOL = 1e-4


def test_sample_regular():
    # Create the object
    smote = SMOTETomek(random_state=RND_SEED)
    # Fit the data
    smote.fit(X, Y)

    X_resampled, y_resampled = smote.fit_sample(X, Y)

    X_gt = np.array([[0.20622591, 0.0582794], [0.68481731, 0.51935141],
                     [1.34192108, -0.13367336], [0.62366841, -0.21312976],
                     [1.61091956, -0.40283504], [-0.37162401, -2.19400981],
                     [0.74680821, 1.63827342], [0.61472253, -0.82309052],
                     [0.19893132, -0.47761769], [0.97407872, 0.44454207],
                     [1.40301027, -0.83648734], [-1.20515198, -1.02689695],
                     [-0.23374509, 0.18370049], [-0.32635887, -0.29299653],
                     [-0.00288378, 0.84259929], [1.79580611, -0.02219234],
                     [0.38307743, -0.05670439], [0.93976473, -0.06570176],
                     [0.70319159, -0.02571668], [0.75052536, -0.19246517]])
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_regular_half():
    # Create the object
    ratio = 0.8
    smote = SMOTETomek(ratio=ratio, random_state=RND_SEED)
    # Fit the data
    smote.fit(X, Y)

    X_resampled, y_resampled = smote.fit_sample(X, Y)

    X_gt = np.array([[0.20622591, 0.0582794], [0.68481731, 0.51935141],
                     [1.34192108, -0.13367336], [0.62366841, -0.21312976],
                     [1.61091956, -0.40283504], [-0.37162401, -2.19400981],
                     [0.74680821, 1.63827342], [0.61472253, -0.82309052],
                     [0.19893132, -0.47761769], [0.97407872, 0.44454207],
                     [1.40301027, -0.83648734], [-1.20515198, -1.02689695],
                     [-0.23374509, 0.18370049], [-0.32635887, -0.29299653],
                     [-0.00288378, 0.84259929], [1.79580611, -0.02219234],
                     [0.45784496, -0.1053161]])
    y_gt = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_validate_estimator_init():
    # Create a SMOTE and Tomek object
    smote = SMOTE(random_state=RND_SEED)
    tomek = TomekLinks(random_state=RND_SEED)

    smt = SMOTETomek(smote=smote, tomek=tomek, random_state=RND_SEED)

    X_resampled, y_resampled = smt.fit_sample(X, Y)

    X_gt = np.array([[0.20622591, 0.0582794], [0.68481731, 0.51935141],
                     [1.34192108, -0.13367336], [0.62366841, -0.21312976],
                     [1.61091956, -0.40283504], [-0.37162401, -2.19400981],
                     [0.74680821, 1.63827342], [0.61472253, -0.82309052],
                     [0.19893132, -0.47761769], [0.97407872, 0.44454207],
                     [1.40301027, -0.83648734], [-1.20515198, -1.02689695],
                     [-0.23374509, 0.18370049], [-0.32635887, -0.29299653],
                     [-0.00288378, 0.84259929], [1.79580611, -0.02219234],
                     [0.38307743, -0.05670439], [0.93976473, -0.06570176],
                     [0.70319159, -0.02571668], [0.75052536, -0.19246517]])
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_validate_estimator_default():
    smt = SMOTETomek(random_state=RND_SEED)

    X_resampled, y_resampled = smt.fit_sample(X, Y)

    X_gt = np.array([[0.20622591, 0.0582794], [0.68481731, 0.51935141],
                     [1.34192108, -0.13367336], [0.62366841, -0.21312976],
                     [1.61091956, -0.40283504], [-0.37162401, -2.19400981],
                     [0.74680821, 1.63827342], [0.61472253, -0.82309052],
                     [0.19893132, -0.47761769], [0.97407872, 0.44454207],
                     [1.40301027, -0.83648734], [-1.20515198, -1.02689695],
                     [-0.23374509, 0.18370049], [-0.32635887, -0.29299653],
                     [-0.00288378, 0.84259929], [1.79580611, -0.02219234],
                     [0.38307743, -0.05670439], [0.93976473, -0.06570176],
                     [0.70319159, -0.02571668], [0.75052536, -0.19246517]])
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_validate_estimator_deprecation():
    X_gt = np.array([[0.20622591, 0.0582794], [0.68481731, 0.51935141],
                     [1.34192108, -0.13367336], [0.62366841, -0.21312976],
                     [1.61091956, -0.40283504], [-0.37162401, -2.19400981],
                     [0.74680821, 1.63827342], [0.61472253, -0.82309052],
                     [0.19893132, -0.47761769], [0.97407872, 0.44454207],
                     [1.40301027, -0.83648734], [-1.20515198, -1.02689695],
                     [-0.23374509, 0.18370049], [-0.32635887, -0.29299653],
                     [-0.00288378, 0.84259929], [1.79580611, -0.02219234],
                     [0.38307743, -0.05670439], [0.93976473, -0.06570176],
                     [0.70319159, -0.02571668], [0.75052536, -0.19246517]])
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0])

    smt = SMOTETomek(random_state=RND_SEED, n_jobs=-1)
    X_resampled, y_resampled = smt.fit_sample(X, Y)
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)

    smt = SMOTETomek(random_state=RND_SEED, k=5)
    X_resampled, y_resampled = smt.fit_sample(X, Y)
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_error_wrong_object():
    # Create a SMOTE and Tomek object
    smote = 'rnd'
    tomek = 'rnd'

    smt = SMOTETomek(smote=smote, random_state=RND_SEED)
    assert_raises_regex(ValueError, "smote needs to be a SMOTE",
                        smt.fit, X, Y)
    smt = SMOTETomek(tomek=tomek, random_state=RND_SEED)
    assert_raises_regex(ValueError, "tomek needs to be a TomekLinks",
                        smt.fit, X, Y)
