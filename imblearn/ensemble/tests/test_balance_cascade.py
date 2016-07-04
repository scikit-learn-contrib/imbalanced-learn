"""Test the module balance cascade."""
from __future__ import print_function

import os

import numpy as np
from numpy.testing import assert_raises
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_warns

from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import check_estimator

from imblearn.ensemble import BalanceCascade

# Generate a global dataset to use
RND_SEED = 0
X, Y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=5000, random_state=RND_SEED)


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
    assert_warns(RuntimeWarning, bc.fit, X, y_single_class)


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
    assert_equal(bc.stats_c_[0], 500)
    assert_equal(bc.stats_c_[1], 4500)


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
    bc = BalanceCascade(ratio=ratio, random_state=RND_SEED,
                        return_indices=True)

    # Get the different subset
    X_resampled, y_resampled, idx_under = bc.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'bc_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'bc_y.npy'))
    idx_gt = np.load(os.path.join(currdir, 'data', 'bc_idx.npy'))
    # Check each array
    for idx in range(X_gt.size):
        assert_array_equal(X_resampled[idx], X_gt[idx])
        assert_array_equal(y_resampled[idx], y_gt[idx])
        assert_array_equal(idx_under[idx], idx_gt[idx])


def test_fit_sample_half():
    """Test the fit and sample routine with 0.5 ratio."""

    # Define the ratio parameter
    ratio = 0.5

    # Create the sampling object
    bc = BalanceCascade(ratio=ratio, random_state=RND_SEED)

    # Get the different subset
    X_resampled, y_resampled = bc.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'bc_x_05.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'bc_y_05.npy'))
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
    bc = BalanceCascade(ratio=ratio, random_state=RND_SEED,
                        return_indices=True, classifier=classifier)

    # Get the different subset
    X_resampled, y_resampled, idx_under = bc.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'bc_x_dt.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'bc_y_dt.npy'))
    idx_gt = np.load(os.path.join(currdir, 'data', 'bc_idx_dt.npy'))
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
    bc = BalanceCascade(ratio=ratio, random_state=RND_SEED,
                        return_indices=True, classifier=classifier)

    # Get the different subset
    X_resampled, y_resampled, idx_under = bc.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'bc_x_rf.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'bc_y_rf.npy'))
    idx_gt = np.load(os.path.join(currdir, 'data', 'bc_idx_rf.npy'))
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
    bc = BalanceCascade(ratio=ratio, random_state=RND_SEED,
                        return_indices=True, classifier=classifier)

    # Get the different subset
    X_resampled, y_resampled, idx_under = bc.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'bc_x_adb.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'bc_y_adb.npy'))
    idx_gt = np.load(os.path.join(currdir, 'data', 'bc_idx_adb.npy'))
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
    bc = BalanceCascade(ratio=ratio, random_state=RND_SEED,
                        return_indices=True, classifier=classifier)

    # Get the different subset
    X_resampled, y_resampled, idx_under = bc.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'bc_x_gb.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'bc_y_gb.npy'))
    idx_gt = np.load(os.path.join(currdir, 'data', 'bc_idx_gb.npy'))
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
    bc = BalanceCascade(ratio=ratio, random_state=RND_SEED,
                        return_indices=True, classifier=classifier)

    # Get the different subset
    X_resampled, y_resampled, idx_under = bc.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'bc_x_svm.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'bc_y_svm.npy'))
    idx_gt = np.load(os.path.join(currdir, 'data', 'bc_idx_svm.npy'))
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
    """Test the fit and sample routine with auto ratio with a static number
    of subsets."""

    # Define the ratio parameter
    ratio = 'auto'
    n_subset = 4

    # Create the sampling object
    bc = BalanceCascade(ratio=ratio, random_state=RND_SEED,
                        return_indices=True, n_max_subset=n_subset)

    # Get the different subset
    X_resampled, y_resampled, idx_under = bc.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'bc_x_n_sub.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'bc_y_n_sub.npy'))
    idx_gt = np.load(os.path.join(currdir, 'data', 'bc_idx_n_sub.npy'))
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
    assert_raises(RuntimeError, bc.sample, np.random.random((100, 40)),
                  np.array([0] * 50 + [1] * 50))
