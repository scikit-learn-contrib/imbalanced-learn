"""Test the module easy ensemble."""
from __future__ import print_function

import numpy as np
from numpy.testing import (assert_array_equal, assert_equal, assert_raises,
                           assert_warns)
from sklearn.utils.estimator_checks import check_estimator

from imblearn.ensemble import EasyEnsemble

# Generate a global dataset to use
RND_SEED = 0
X = np.array([[0.5220963, 0.11349303], [0.59091459, 0.40692742],
              [1.10915364, 0.05718352], [0.22039505, 0.26469445],
              [1.35269503, 0.44812421], [0.85117925, 1.0185556],
              [-2.10724436, 0.70263997], [-0.23627356, 0.30254174],
              [-1.23195149, 0.15427291], [-0.58539673, 0.62515052]])
Y = np.array([1, 2, 2, 2, 1, 0, 1, 1, 1, 0])


def test_ee_sk_estimator():
    """Test the sklearn estimator compatibility"""
    check_estimator(EasyEnsemble)


def test_ee_bad_ratio():
    """Test either if an error is raised with a wrong decimal value for
    the ratio"""

    # Define a negative ratio
    ratio = -1.0
    ee = EasyEnsemble(ratio=ratio)
    assert_raises(ValueError, ee.fit, X, Y)

    # Define a ratio greater than 1
    ratio = 100.0
    ee = EasyEnsemble(ratio=ratio)
    assert_raises(ValueError, ee.fit, X, Y)

    # Define ratio as an unknown string
    ratio = 'rnd'
    ee = EasyEnsemble(ratio=ratio)
    assert_raises(ValueError, ee.fit, X, Y)

    # Define ratio as a list which is not supported
    ratio = [.5, .5]
    ee = EasyEnsemble(ratio=ratio)
    assert_raises(ValueError, ee.fit, X, Y)


def test_ee_init():
    """Test the initialisation of the object"""

    # Define a ratio
    ratio = 1.
    ee = EasyEnsemble(ratio=ratio, random_state=RND_SEED)

    assert_equal(ee.ratio, ratio)
    assert_equal(ee.replacement, False)
    assert_equal(ee.n_subsets, 10)
    assert_equal(ee.random_state, RND_SEED)


def test_ee_fit_single_class():
    """Test either if an error when there is a single class"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    ee = EasyEnsemble(ratio=ratio, random_state=RND_SEED)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_warns(UserWarning, ee.fit, X, y_single_class)


def test_ee_fit_invalid_ratio():
    """Test either if an error is raised when the balancing ratio to fit is
    smaller than the one of the data"""

    # Create the object
    ratio = 1. / 10000.
    ee = EasyEnsemble(ratio=ratio, random_state=RND_SEED)
    # Fit the data
    assert_raises(RuntimeError, ee.fit, X, Y)


def test_ee_fit():
    """Test the fitting method"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    ee = EasyEnsemble(ratio=ratio, random_state=RND_SEED)
    # Fit the data
    ee.fit(X, Y)

    # Check if the data information have been computed
    assert_equal(ee.min_c_, 0)
    assert_equal(ee.maj_c_, 1)
    assert_equal(ee.stats_c_[0], 2)
    assert_equal(ee.stats_c_[2], 3)
    assert_equal(ee.stats_c_[1], 5)


def test_sample_wt_fit():
    """Test either if an error is raised when sample is called before
    fitting"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    ee = EasyEnsemble(ratio=ratio, random_state=RND_SEED)
    assert_raises(RuntimeError, ee.sample, X, Y)


def test_fit_sample_auto():
    """Test the fit and sample routine with auto ratio."""

    # Define the ratio parameter
    ratio = 'auto'

    # Create the sampling object
    ee = EasyEnsemble(
        ratio=ratio, random_state=RND_SEED, return_indices=True, n_subsets=3)

    # Get the different subset
    X_resampled, y_resampled, idx_under = ee.fit_sample(X, Y)

    X_gt = np.array([[[0.85117925, 1.0185556], [-0.58539673, 0.62515052],
                      [1.35269503, 0.44812421], [0.5220963, 0.11349303],
                      [1.10915364, 0.05718352], [0.22039505, 0.26469445]],
                     [[0.85117925, 1.0185556], [-0.58539673, 0.62515052],
                      [-1.23195149, 0.15427291], [-2.10724436, 0.70263997],
                      [0.22039505, 0.26469445], [1.10915364, 0.05718352]],
                     [[0.85117925, 1.0185556], [-0.58539673, 0.62515052],
                      [-1.23195149, 0.15427291], [0.5220963, 0.11349303],
                      [1.10915364, 0.05718352], [0.59091459, 0.40692742]]])
    y_gt = np.array([[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2],
                     [0, 0, 1, 1, 2, 2]])
    idx_gt = np.array([[5, 9, 4, 0, 2, 3], [5, 9, 8, 6, 3, 2],
                       [5, 9, 8, 0, 2, 1]])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


def test_fit_sample_half():
    """Test the fit and sample routine with 0.5 ratio."""

    # Define the ratio parameter
    ratio = 0.6

    # Create the sampling object
    ee = EasyEnsemble(ratio=ratio, random_state=RND_SEED, n_subsets=3)

    # Get the different subset
    X_resampled, y_resampled = ee.fit_sample(X, Y)

    X_gt = np.array([[[0.85117925, 1.0185556], [-0.58539673, 0.62515052],
                      [1.35269503, 0.44812421], [0.5220963, 0.11349303],
                      [-2.10724436, 0.70263997], [1.10915364, 0.05718352],
                      [0.22039505, 0.26469445], [0.59091459, 0.40692742]],
                     [[0.85117925, 1.0185556], [-0.58539673, 0.62515052],
                      [-1.23195149, 0.15427291], [-2.10724436, 0.70263997],
                      [0.5220963, 0.11349303], [0.22039505, 0.26469445],
                      [1.10915364, 0.05718352], [0.59091459, 0.40692742]],
                     [[0.85117925, 1.0185556], [-0.58539673, 0.62515052],
                      [-1.23195149, 0.15427291], [0.5220963, 0.11349303],
                      [1.35269503, 0.44812421], [1.10915364, 0.05718352],
                      [0.59091459, 0.40692742], [0.22039505, 0.26469445]]])
    y_gt = np.array([[0, 0, 1, 1, 1, 2, 2, 2], [0, 0, 1, 1, 1, 2, 2, 2],
                     [0, 0, 1, 1, 1, 2, 2, 2]])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_random_state_none():
    """Test that the processing is going throw with random state being None."""

    # Define the ratio parameter
    ratio = 'auto'

    # Create the sampling object
    ee = EasyEnsemble(ratio=ratio, random_state=None)

    # Get the different subset
    X_resampled, y_resampled = ee.fit_sample(X, Y)


def test_sample_wrong_X():
    """Test either if an error is raised when X is different at fitting
    and sampling"""

    # Create the object
    ee = EasyEnsemble(random_state=RND_SEED)
    ee.fit(X, Y)
    assert_raises(RuntimeError, ee.sample,
                  np.random.random((100, 40)), np.array([0] * 50 + [1] * 50))


def test_continuous_error():
    """Test either if an error is raised when the target are continuous
    type"""

    # continuous case
    y = np.linspace(0, 1, 10)
    ee = EasyEnsemble(random_state=RND_SEED)
    assert_warns(UserWarning, ee.fit, X, y)
