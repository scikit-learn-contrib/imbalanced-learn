"""Test the module neighbourhood cleaning rule."""
from __future__ import print_function

import numpy as np
from numpy.testing import (assert_array_equal, assert_equal, assert_raises,
                           assert_warns)
from sklearn.utils.estimator_checks import check_estimator
from sklearn.neighbors import NearestNeighbors

from imblearn.under_sampling import NeighbourhoodCleaningRule

# Generate a global dataset to use
RND_SEED = 0
X = np.array([[1.57737838, 0.1997882], [0.8960075, 0.46130762],
              [0.34096173, 0.50947647], [-0.91735824, 0.93110278],
              [-0.14619583, 1.33009918], [-0.20413357, 0.64628718],
              [0.85713638, 0.91069295], [0.35967591, 2.61186964],
              [0.43142011, 0.52323596], [0.90701028, -0.57636928],
              [-1.20809175, -1.49917302], [-0.60497017, -0.66630228],
              [1.39272351, -0.51631728], [-1.55581933, 1.09609604],
              [1.55157493, -1.6981518]])
Y = np.array([1, 2, 1, 1, 2, 1, 2, 2, 1, 2, 0, 0, 2, 1, 2])


def test_ncr_sk_estimator():
    """Test the sklearn estimator compatibility"""
    check_estimator(NeighbourhoodCleaningRule)


def test_ncr_init():
    """Test the initialisation of the object"""

    # Define a ratio
    ncr = NeighbourhoodCleaningRule(random_state=RND_SEED)

    assert_equal(ncr.n_neighbors, 3)
    assert_equal(ncr.n_jobs, 1)
    assert_equal(ncr.random_state, RND_SEED)


def test_ncr_fit_single_class():
    """Test either if an error when there is a single class"""

    # Create the object
    ncr = NeighbourhoodCleaningRule(random_state=RND_SEED)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_warns(UserWarning, ncr.fit, X, y_single_class)


def test_ncr_fit():
    """Test the fitting method"""

    # Create the object
    ncr = NeighbourhoodCleaningRule(random_state=RND_SEED)
    # Fit the data
    ncr.fit(X, Y)

    # Check if the data information have been computed
    assert_equal(ncr.min_c_, 0)
    assert_equal(ncr.maj_c_, 2)
    assert_equal(ncr.stats_c_[0], 2)
    assert_equal(ncr.stats_c_[1], 6)
    assert_equal(ncr.stats_c_[2], 7)


def test_ncr_sample_wt_fit():
    """Test either if an error is raised when sample is called before
    fitting"""

    # Create the object
    ncr = NeighbourhoodCleaningRule(random_state=RND_SEED)
    assert_raises(RuntimeError, ncr.sample, X, Y)


def test_ncr_fit_sample():
    """Test the fit sample routine"""

    # Resample the data
    ncr = NeighbourhoodCleaningRule(random_state=RND_SEED)
    X_resampled, y_resampled = ncr.fit_sample(X, Y)

    X_gt = np.array([[-1.20809175, -1.49917302], [-0.60497017, -0.66630228],
                     [-0.91735824, 0.93110278], [-0.20413357, 0.64628718],
                     [0.35967591, 2.61186964], [-1.55581933, 1.09609604],
                     [1.55157493, -1.6981518]])
    y_gt = np.array([0, 0, 1, 1, 2, 1, 2])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_ncr_fit_sample_with_indices():
    """Test the fit sample routine with indices support"""

    # Resample the data
    ncr = NeighbourhoodCleaningRule(return_indices=True, random_state=RND_SEED)
    X_resampled, y_resampled, idx_under = ncr.fit_sample(X, Y)

    X_gt = np.array([[-1.20809175, -1.49917302], [-0.60497017, -0.66630228],
                     [-0.91735824, 0.93110278], [-0.20413357, 0.64628718],
                     [0.35967591, 2.61186964], [-1.55581933, 1.09609604],
                     [1.55157493, -1.6981518]])
    y_gt = np.array([0, 0, 1, 1, 2, 1, 2])
    idx_gt = np.array([10, 11, 3, 5, 7, 13, 14])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


def test_ncr_sample_wrong_X():
    """Test either if an error is raised when X is different at fitting
    and sampling"""

    # Create the object
    ncr = NeighbourhoodCleaningRule(random_state=RND_SEED)
    ncr.fit(X, Y)
    assert_raises(RuntimeError, ncr.sample,
                  np.random.random((100, 40)), np.array([0] * 50 + [1] * 50))


def test_continuous_error():
    """Test either if an error is raised when the target are continuous
    type"""

    # continuous case
    y = np.linspace(0, 1, 15)
    ncr = NeighbourhoodCleaningRule(random_state=RND_SEED)
    assert_warns(UserWarning, ncr.fit, X, y)


def test_ncr_fit_sample_nn_obj():
    """Test fit-sample with nn object"""

    # Resample the data
    nn = NearestNeighbors(n_neighbors=3)
    ncr = NeighbourhoodCleaningRule(
        return_indices=True, random_state=RND_SEED, n_neighbors=nn)
    X_resampled, y_resampled, idx_under = ncr.fit_sample(X, Y)

    X_gt = np.array([[-1.20809175, -1.49917302], [-0.60497017, -0.66630228],
                     [-0.91735824, 0.93110278], [-0.20413357, 0.64628718],
                     [0.35967591, 2.61186964], [-1.55581933, 1.09609604],
                     [1.55157493, -1.6981518]])
    y_gt = np.array([0, 0, 1, 1, 2, 1, 2])
    idx_gt = np.array([10, 11, 3, 5, 7, 13, 14])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


def test_ncr_wrong_nn_obj():
    """Test either if an error is raised with wrong NN object"""

    # Resample the data
    nn = 'rnd'
    ncr = NeighbourhoodCleaningRule(
        return_indices=True, random_state=RND_SEED, n_neighbors=nn)
    assert_raises(ValueError, ncr.fit_sample, X, Y)
