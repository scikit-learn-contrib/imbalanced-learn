"""Test the module balance cascade."""
from __future__ import print_function

import numpy as np
from numpy.testing import (assert_array_equal, assert_equal, assert_raises,
                           assert_warns)
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.estimator_checks import check_estimator

from imblearn.ensemble import BalanceCascade

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


def test_bc_sk_estimator():
    """Test the sklearn estimator compatibility"""
    check_estimator(BalanceCascade)


def test_bc_bad_ratio():
    """Test either if an error is raised with a wrong decimal value for
    the ratio"""

    # Define a negative ratio
    ratio = -1.0
    bc = BalanceCascade(ratio=ratio)
    assert_raises(ValueError, bc.fit, X, Y)

    # Define a ratio greater than 1
    ratio = 100.0
    bc = BalanceCascade(ratio=ratio)
    assert_raises(ValueError, bc.fit, X, Y)

    # Define ratio as an unknown string
    ratio = 'rnd'
    bc = BalanceCascade(ratio=ratio)
    assert_raises(ValueError, bc.fit, X, Y)

    # Define ratio as a list which is not supported
    ratio = [.5, .5]
    bc = BalanceCascade(ratio=ratio)
    assert_raises(ValueError, bc.fit, X, Y)


def test_bc_init():
    """Test the initialisation of the object"""

    # Define a ratio
    ratio = 1.
    bc = BalanceCascade(ratio=ratio, random_state=RND_SEED)

    assert_equal(bc.ratio, ratio)
    assert_equal(bc.bootstrap, True)
    assert_equal(bc.n_max_subset, None)
    assert_equal(bc.random_state, RND_SEED)


def test_bc_fit_single_class():
    """Test either if an error when there is a single class"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    bc = BalanceCascade(ratio=ratio, random_state=RND_SEED)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_warns(UserWarning, bc.fit, X, y_single_class)


def test_bc_fit_invalid_ratio():
    """Test either if an error is raised when the balancing ratio to fit is
    smaller than the one of the data"""

    # Create the object
    ratio = 1. / 10000.
    bc = BalanceCascade(ratio=ratio, random_state=RND_SEED)
    # Fit the data
    assert_raises(RuntimeError, bc.fit_sample, X, Y)


def test_bc_fit():
    """Test the fitting method"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    bc = BalanceCascade(ratio=ratio, random_state=RND_SEED)
    # Fit the data
    bc.fit(X, Y)

    # Check if the data information have been computed
    assert_equal(bc.min_c_, 0)
    assert_equal(bc.maj_c_, 1)
    assert_equal(bc.stats_c_[0], 8)
    assert_equal(bc.stats_c_[1], 12)


def test_sample_wt_fit():
    """Test either if an error is raised when sample is called before
    fitting"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    bc = BalanceCascade(ratio=ratio, random_state=RND_SEED)
    assert_raises(RuntimeError, bc.sample, X, Y)


def test_fit_sample_auto():
    """Test the fit and sample routine with auto ratio."""

    # Define the ratio parameter
    ratio = 'auto'

    # Create the sampling object
    bc = BalanceCascade(
        ratio=ratio,
        random_state=RND_SEED,
        return_indices=True,
        bootstrap=False)

    # Get the different subset
    X_resampled, y_resampled, idx_under = bc.fit_sample(X, Y)

    X_gt = np.array(
        [
            np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                      [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                      [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                      [1.15514042, 0.0129463], [0.08711622, 0.93259929],
                      [0.70472253, -0.73309052], [-0.14374509, 0.27370049],
                      [0.83680821, 1.72827342], [-0.18410027, -0.45194484],
                      [-0.28162401, -2.10400981], [-1.11515198, -0.93689695]]),
            np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                      [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                      [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                      [1.15514042, 0.0129463], [0.70472253, -0.73309052],
                      [-0.18410027, -0.45194484], [0.77481731, 0.60935141],
                      [0.3084254, 0.33299982], [0.28893132, -0.38761769],
                      [0.9281014, 0.53085498]])
        ],
        dtype=object)
    y_gt = np.array(
        [
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        ],
        dtype=object)
    idx_gt = np.array(
        [
            np.array([0, 2, 3, 4, 11, 12, 17, 19, 10, 18, 8, 16, 6, 14, 5,
                      13]),
            np.array([0, 2, 3, 4, 11, 12, 17, 19, 10, 8, 14, 1, 7, 9, 15])
        ],
        dtype=object)
    # Check each array
    for idx in range(X_gt.size):
        assert_array_equal(X_resampled[idx], X_gt[idx])
        assert_array_equal(y_resampled[idx], y_gt[idx])
        assert_array_equal(idx_under[idx], idx_gt[idx])


def test_fit_sample_half():
    """Test the fit and sample routine with 0.5 ratio."""

    # Define the ratio parameter
    ratio = 0.8

    # Create the sampling object
    bc = BalanceCascade(ratio=ratio, random_state=RND_SEED, bootstrap=False)

    # Get the different subset
    X_resampled, y_resampled = bc.fit_sample(X, Y)

    X_gt = np.array(
        [
            np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                      [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                      [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                      [1.15514042, 0.0129463], [0.08711622, 0.93259929],
                      [0.70472253, -0.73309052], [-0.14374509, 0.27370049],
                      [0.83680821, 1.72827342], [-0.18410027, -0.45194484],
                      [-0.28162401, -2.10400981], [-1.11515198, -0.93689695],
                      [0.9281014, 0.53085498], [0.3084254, 0.33299982]]),
            np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                      [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                      [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                      [1.15514042, 0.0129463], [0.70472253, -0.73309052],
                      [-0.18410027, -0.45194484], [0.77481731, 0.60935141],
                      [0.28893132, -0.38761769]])
        ],
        dtype=object)

    y_gt = np.array(
        [
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        ],
        dtype=object)
    # Check each array
    for idx in range(X_gt.size):
        assert_array_equal(X_resampled[idx], X_gt[idx])
        assert_array_equal(y_resampled[idx], y_gt[idx])


def test_fit_sample_auto_decision_tree():
    """Test the fit and sample routine with auto ratio with a decision
    tree."""

    # Define the ratio parameter
    ratio = 'auto'
    classifier = 'decision-tree'

    # Create the sampling object
    bc = BalanceCascade(
        ratio=ratio,
        random_state=RND_SEED,
        return_indices=True,
        classifier=classifier)

    # Get the different subset
    X_resampled, y_resampled, idx_under = bc.fit_sample(X, Y)

    X_gt = np.array(
        [
            np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                      [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                      [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                      [1.15514042, 0.0129463], [0.08711622, 0.93259929],
                      [0.70472253, -0.73309052], [-0.14374509, 0.27370049],
                      [0.83680821, 1.72827342], [-0.18410027, -0.45194484],
                      [-0.28162401, -2.10400981], [-1.11515198, -0.93689695]]),
            np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                      [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                      [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                      [-1.11515198, -0.93689695], [0.77481731, 0.60935141],
                      [0.3084254, 0.33299982], [0.28893132, -0.38761769],
                      [0.9281014, 0.53085498]])
        ],
        dtype=object)
    y_gt = np.array(
        [
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        ],
        dtype=object)
    idx_gt = np.array(
        [
            np.array([0, 2, 3, 4, 11, 12, 17, 19, 10, 18, 8, 16, 6, 14, 5,
                      13]),
            np.array([0, 2, 3, 4, 11, 12, 17, 19, 13, 1, 7, 9, 15])
        ],
        dtype=object)
    # Check each array
    for idx in range(X_gt.size):
        assert_array_equal(X_resampled[idx], X_gt[idx])
        assert_array_equal(y_resampled[idx], y_gt[idx])
        assert_array_equal(idx_under[idx], idx_gt[idx])


def test_fit_sample_auto_random_forest():
    """Test the fit and sample routine with auto ratio with a random
    forest."""

    # Define the ratio parameter
    ratio = 'auto'
    classifier = 'random-forest'

    # Create the sampling object
    bc = BalanceCascade(
        ratio=ratio,
        random_state=RND_SEED,
        return_indices=True,
        classifier=classifier)

    # Get the different subset
    X_resampled, y_resampled, idx_under = bc.fit_sample(X, Y)

    X_gt = np.array(
        [
            np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                      [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                      [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                      [1.15514042, 0.0129463], [0.08711622, 0.93259929],
                      [0.70472253, -0.73309052], [-0.14374509, 0.27370049],
                      [0.83680821, 1.72827342], [-0.18410027, -0.45194484],
                      [-0.28162401, -2.10400981], [-1.11515198, -0.93689695]]),
            np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                      [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                      [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                      [1.15514042, 0.0129463], [-0.14374509, 0.27370049],
                      [-1.11515198, -0.93689695], [0.77481731, 0.60935141],
                      [0.3084254, 0.33299982], [0.28893132, -0.38761769],
                      [0.9281014, 0.53085498]])
        ],
        dtype=object)
    y_gt = np.array(
        [
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        ],
        dtype=object)
    idx_gt = np.array(
        [
            np.array([0, 2, 3, 4, 11, 12, 17, 19, 10, 18, 8, 16, 6, 14, 5,
                      13]),
            np.array([0, 2, 3, 4, 11, 12, 17, 19, 10, 16, 13, 1, 7, 9, 15])
        ],
        dtype=object)
    # Check each array
    for idx in range(X_gt.size):
        assert_array_equal(X_resampled[idx], X_gt[idx])
        assert_array_equal(y_resampled[idx], y_gt[idx])
        assert_array_equal(idx_under[idx], idx_gt[idx])


def test_fit_sample_auto_adaboost():
    """Test the fit and sample routine with auto ratio with a adaboost."""

    # Define the ratio parameter
    ratio = 'auto'
    classifier = 'adaboost'

    # Create the sampling object
    bc = BalanceCascade(
        ratio=ratio,
        random_state=RND_SEED,
        return_indices=True,
        classifier=classifier)

    # Get the different subset
    X_resampled, y_resampled, idx_under = bc.fit_sample(X, Y)

    X_gt = np.array(
        [
            np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                      [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                      [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                      [1.15514042, 0.0129463], [0.08711622, 0.93259929],
                      [0.70472253, -0.73309052], [-0.14374509, 0.27370049],
                      [0.83680821, 1.72827342], [-0.18410027, -0.45194484],
                      [-0.28162401, -2.10400981], [-1.11515198, -0.93689695]]),
            np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                      [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                      [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                      [-0.14374509, 0.27370049], [-1.11515198, -0.93689695],
                      [0.77481731, 0.60935141], [0.3084254, 0.33299982],
                      [0.28893132, -0.38761769], [0.9281014, 0.53085498]])
        ],
        dtype=object)
    y_gt = np.array(
        [
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        ],
        dtype=object)
    idx_gt = np.array(
        [
            np.array([0, 2, 3, 4, 11, 12, 17, 19, 10, 18, 8, 16, 6, 14, 5,
                      13]),
            np.array([0, 2, 3, 4, 11, 12, 17, 19, 16, 13, 1, 7, 9, 15])
        ],
        dtype=object)
    # Check each array
    for idx in range(X_gt.size):
        assert_array_equal(X_resampled[idx], X_gt[idx])
        assert_array_equal(y_resampled[idx], y_gt[idx])
        assert_array_equal(idx_under[idx], idx_gt[idx])


def test_fit_sample_auto_gradient_boosting():
    """Test the fit and sample routine with auto ratio with a gradient
    boosting."""

    # Define the ratio parameter
    ratio = 'auto'
    classifier = 'gradient-boosting'

    # Create the sampling object
    bc = BalanceCascade(
        ratio=ratio,
        random_state=RND_SEED,
        return_indices=True,
        classifier=classifier)

    # Get the different subset
    X_resampled, y_resampled, idx_under = bc.fit_sample(X, Y)

    X_gt = np.array(
        [
            np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                      [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                      [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                      [1.15514042, 0.0129463], [0.08711622, 0.93259929],
                      [0.70472253, -0.73309052], [-0.14374509, 0.27370049],
                      [0.83680821, 1.72827342], [-0.18410027, -0.45194484],
                      [-0.28162401, -2.10400981], [-1.11515198, -0.93689695]]),
            np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                      [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                      [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                      [-0.14374509, 0.27370049], [-1.11515198, -0.93689695],
                      [0.77481731, 0.60935141], [0.3084254, 0.33299982],
                      [0.28893132, -0.38761769], [0.9281014, 0.53085498]])
        ],
        dtype=object)
    y_gt = np.array(
        [
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        ],
        dtype=object)
    idx_gt = np.array(
        [
            np.array([0, 2, 3, 4, 11, 12, 17, 19, 10, 18, 8, 16, 6, 14, 5,
                      13]),
            np.array([0, 2, 3, 4, 11, 12, 17, 19, 16, 13, 1, 7, 9, 15])
        ],
        dtype=object)

    # Check each array
    for idx in range(X_gt.size):
        assert_array_equal(X_resampled[idx], X_gt[idx])
        assert_array_equal(y_resampled[idx], y_gt[idx])
        assert_array_equal(idx_under[idx], idx_gt[idx])


def test_fit_sample_auto_linear_svm():
    """Test the fit and sample routine with auto ratio with a linear
    svm."""

    # Define the ratio parameter
    ratio = 'auto'
    classifier = 'linear-svm'

    # Create the sampling object
    bc = BalanceCascade(
        ratio=ratio,
        random_state=RND_SEED,
        return_indices=True,
        classifier=classifier)

    # Get the different subset
    X_resampled, y_resampled, idx_under = bc.fit_sample(X, Y)

    X_gt = np.array(
        [
            np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                      [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                      [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                      [1.15514042, 0.0129463], [0.08711622, 0.93259929],
                      [0.70472253, -0.73309052], [-0.14374509, 0.27370049],
                      [0.83680821, 1.72827342], [-0.18410027, -0.45194484],
                      [-0.28162401, -2.10400981], [-1.11515198, -0.93689695]]),
            np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                      [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                      [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                      [1.15514042, 0.0129463], [0.70472253, -0.73309052],
                      [0.77481731, 0.60935141], [0.3084254, 0.33299982],
                      [0.28893132, -0.38761769], [0.9281014, 0.53085498]])
        ],
        dtype=object)
    y_gt = np.array(
        [
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        ],
        dtype=object)
    idx_gt = np.array(
        [
            np.array([0, 2, 3, 4, 11, 12, 17, 19, 10, 18, 8, 16, 6, 14, 5,
                      13]),
            np.array([0, 2, 3, 4, 11, 12, 17, 19, 10, 8, 1, 7, 9, 15])
        ],
        dtype=object)

    # Check each array
    for idx in range(X_gt.size):
        assert_array_equal(X_resampled[idx], X_gt[idx])
        assert_array_equal(y_resampled[idx], y_gt[idx])
        assert_array_equal(idx_under[idx], idx_gt[idx])


def test_init_wrong_classifier():
    """Test either if an error is raised the classifier provided is unknown."""

    # Define the ratio parameter
    classifier = 'rnd'

    bc = BalanceCascade(classifier=classifier)
    assert_raises(NotImplementedError, bc.fit_sample, X, Y)


def test_fit_sample_auto_early_stop():
    """Test the fit and sample routine with auto ratio with 1 subset."""

    # Define the ratio parameter
    ratio = 'auto'
    n_subset = 1

    # Create the sampling object
    bc = BalanceCascade(
        ratio=ratio,
        random_state=RND_SEED,
        return_indices=True,
        n_max_subset=n_subset)

    # Get the different subset
    X_resampled, y_resampled, idx_under = bc.fit_sample(X, Y)

    X_gt = np.array([[[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                      [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                      [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                      [1.15514042, 0.0129463], [0.08711622, 0.93259929],
                      [0.70472253, -0.73309052], [-0.14374509, 0.27370049],
                      [0.83680821, 1.72827342], [-0.18410027, -0.45194484],
                      [-0.28162401, -2.10400981], [-1.11515198, -0.93689695]]])

    y_gt = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]])
    idx_gt = np.array(
        [[0, 2, 3, 4, 11, 12, 17, 19, 10, 18, 8, 16, 6, 14, 5, 13]])
    # Check each array
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


def test_fit_sample_auto_early_stop_2():
    """Test the fit and sample routine with auto ratio with a 2 subsets."""

    # Define the ratio parameter
    ratio = 'auto'
    n_subset = 2

    # Create the sampling object
    bc = BalanceCascade(
        ratio=ratio,
        random_state=RND_SEED,
        return_indices=True,
        n_max_subset=n_subset,
        bootstrap=False)

    # Get the different subset
    X_resampled, y_resampled, idx_under = bc.fit_sample(X, Y)

    X_gt = np.array(
        [
            np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                      [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                      [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                      [1.15514042, 0.0129463], [0.08711622, 0.93259929],
                      [0.70472253, -0.73309052], [-0.14374509, 0.27370049],
                      [0.83680821, 1.72827342], [-0.18410027, -0.45194484],
                      [-0.28162401, -2.10400981], [-1.11515198, -0.93689695]]),
            np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                      [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                      [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                      [1.15514042, 0.0129463], [0.70472253, -0.73309052],
                      [-0.18410027, -0.45194484], [0.77481731, 0.60935141],
                      [0.3084254, 0.33299982], [0.28893132, -0.38761769],
                      [0.9281014, 0.53085498]])
        ],
        dtype=object)
    y_gt = np.array(
        [
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        ],
        dtype=object)
    idx_gt = np.array(
        [
            np.array([0, 2, 3, 4, 11, 12, 17, 19, 10, 18, 8, 16, 6, 14, 5,
                      13]),
            np.array([0, 2, 3, 4, 11, 12, 17, 19, 10, 8, 14, 1, 7, 9, 15])
        ],
        dtype=object)
    # Check each array
    for idx in range(X_gt.size):
        assert_array_equal(X_resampled[idx], X_gt[idx])
        assert_array_equal(y_resampled[idx], y_gt[idx])
        assert_array_equal(idx_under[idx], idx_gt[idx])


def test_sample_wrong_X():
    """Test either if an error is raised when X is different at fitting
    and sampling"""

    # Create the object
    bc = BalanceCascade(random_state=RND_SEED)
    bc.fit(X, Y)
    assert_raises(RuntimeError, bc.sample,
                  np.random.random((100, 40)), np.array([0] * 50 + [1] * 50))


def test_multiclass_error():
    """ Test either if an error is raised when the target are not binary
    type. """

    # continuous case
    y = np.linspace(0, 1, 20)
    bc = BalanceCascade(random_state=RND_SEED)
    assert_warns(UserWarning, bc.fit, X, y)

    # multiclass case
    y = np.array([0] * 3 + [1] * 2 + [2] * 15)
    bc = BalanceCascade(random_state=RND_SEED)
    assert_warns(UserWarning, bc.fit, X, y)


def test_give_classifier_obj():
    """Test the fit and sample routine with classifier a object"""

    # Define the ratio parameter
    ratio = 'auto'
    classifier = RandomForestClassifier(random_state=RND_SEED)

    # Create the sampling object
    bc = BalanceCascade(
        ratio=ratio,
        random_state=RND_SEED,
        return_indices=True,
        estimator=classifier)

    # Get the different subset
    X_resampled, y_resampled, idx_under = bc.fit_sample(X, Y)

    X_gt = np.array(
        [
            np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                      [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                      [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                      [1.15514042, 0.0129463], [0.08711622, 0.93259929],
                      [0.70472253, -0.73309052], [-0.14374509, 0.27370049],
                      [0.83680821, 1.72827342], [-0.18410027, -0.45194484],
                      [-0.28162401, -2.10400981], [-1.11515198, -0.93689695]]),
            np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                      [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                      [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                      [1.15514042, 0.0129463], [-0.14374509, 0.27370049],
                      [-1.11515198, -0.93689695], [0.77481731, 0.60935141],
                      [0.3084254, 0.33299982], [0.28893132, -0.38761769],
                      [0.9281014, 0.53085498]])
        ],
        dtype=object)
    y_gt = np.array(
        [
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        ],
        dtype=object)
    idx_gt = np.array(
        [
            np.array([0, 2, 3, 4, 11, 12, 17, 19, 10, 18, 8, 16, 6, 14, 5,
                      13]),
            np.array([0, 2, 3, 4, 11, 12, 17, 19, 10, 16, 13, 1, 7, 9, 15])
        ],
        dtype=object)

    # Check each array
    for idx in range(X_gt.size):
        assert_array_equal(X_resampled[idx], X_gt[idx])
        assert_array_equal(y_resampled[idx], y_gt[idx])
        assert_array_equal(idx_under[idx], idx_gt[idx])


def test_give_classifier_wrong_obj():
    """Test either if an error is raised while a wrong object is passed"""

    # Define the ratio parameter
    ratio = 'auto'
    classifier = 2

    # Create the sampling object
    bc = BalanceCascade(
        ratio=ratio,
        random_state=RND_SEED,
        return_indices=True,
        estimator=classifier)

    # Get the different subset
    assert_raises(ValueError, bc.fit_sample, X, Y)


def test_rf_wth_bootstrap():
    """Test the fit and sample routine with auto ratio with a random
    forest."""

    # Define the ratio parameter
    ratio = 'auto'
    classifier = RandomForestClassifier(random_state=RND_SEED)

    # Create the sampling object
    bc = BalanceCascade(
        ratio=ratio,
        random_state=RND_SEED,
        return_indices=True,
        estimator=classifier,
        bootstrap=False)

    # Get the different subset
    X_resampled, y_resampled, idx_under = bc.fit_sample(X, Y)

    X_gt = np.array(
        [
            np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                      [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                      [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                      [1.15514042, 0.0129463], [0.08711622, 0.93259929],
                      [0.70472253, -0.73309052], [-0.14374509, 0.27370049],
                      [0.83680821, 1.72827342], [-0.18410027, -0.45194484],
                      [-0.28162401, -2.10400981], [-1.11515198, -0.93689695]]),
            np.array([[0.11622591, -0.0317206], [1.25192108, -0.22367336],
                      [0.53366841, -0.30312976], [1.52091956, -0.49283504],
                      [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234],
                      [1.15514042, 0.0129463], [0.77481731, 0.60935141],
                      [0.3084254, 0.33299982], [0.28893132, -0.38761769],
                      [0.9281014, 0.53085498]])
        ],
        dtype=object)
    y_gt = np.array(
        [
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        ],
        dtype=object)
    idx_gt = np.array(
        [
            np.array([0, 2, 3, 4, 11, 12, 17, 19, 10, 18, 8, 16, 6, 14, 5,
                      13]),
            np.array([0, 2, 3, 4, 11, 12, 17, 19, 10, 1, 7, 9, 15])
        ],
        dtype=object)

    # Check each array
    for idx in range(X_gt.size):
        assert_array_equal(X_resampled[idx], X_gt[idx])
        assert_array_equal(y_resampled[idx], y_gt[idx])
        assert_array_equal(idx_under[idx], idx_gt[idx])
