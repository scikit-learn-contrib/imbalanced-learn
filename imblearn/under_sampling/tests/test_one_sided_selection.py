"""Test the module one-sided selection."""
from __future__ import print_function

import numpy as np
from numpy.testing import (assert_array_equal, assert_equal, assert_raises,
                           assert_warns)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.estimator_checks import check_estimator

from imblearn.under_sampling import OneSidedSelection

# Generate a global dataset to use
RND_SEED = 0
X = np.array([[-0.3879569, 0.6894251], [-0.09322739, 1.28177189],
              [-0.77740357, 0.74097941], [0.91542919, -0.65453327],
              [-0.03852113, 0.40910479], [-0.43877303, 1.07366684],
              [-0.85795321, 0.82980738], [-0.18430329, 0.52328473],
              [-0.30126957, -0.66268378], [-0.65571327, 0.42412021],
              [-0.28305528, 0.30284991], [0.20246714, -0.34727125],
              [1.06446472, -1.09279772], [0.30543283, -0.02589502],
              [-0.00717161, 0.00318087]])
Y = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0])


def test_oss_sk_estimator():
    """Test the sklearn estimator compatibility"""
    check_estimator(OneSidedSelection)


def test_oss_init():
    """Test the initialisation of the object"""

    # Define a ratio
    oss = OneSidedSelection(random_state=RND_SEED)

    assert_equal(oss.n_seeds_S, 1)
    assert_equal(oss.n_jobs, 1)
    assert_equal(oss.random_state, RND_SEED)


def test_oss_fit_single_class():
    """Test either if an error when there is a single class"""

    # Create the object
    oss = OneSidedSelection(random_state=RND_SEED)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_warns(UserWarning, oss.fit, X, y_single_class)


def test_oss_fit():
    """Test the fitting method"""

    # Create the object
    oss = OneSidedSelection(random_state=RND_SEED)
    # Fit the data
    oss.fit(X, Y)

    # Check if the data information have been computed
    assert_equal(oss.min_c_, 0)
    assert_equal(oss.maj_c_, 1)
    assert_equal(oss.stats_c_[0], 6)
    assert_equal(oss.stats_c_[1], 9)


def test_oss_sample_wt_fit():
    """Test either if an error is raised when sample is called before
    fitting"""

    # Create the object
    oss = OneSidedSelection(random_state=RND_SEED)
    assert_raises(RuntimeError, oss.sample, X, Y)


def test_oss_fit_sample():
    """Test the fit sample routine"""

    # Resample the data
    oss = OneSidedSelection(random_state=RND_SEED)
    X_resampled, y_resampled = oss.fit_sample(X, Y)

    X_gt = np.array([[-0.3879569, 0.6894251], [0.91542919, -0.65453327],
                     [-0.65571327, 0.42412021], [1.06446472, -1.09279772],
                     [0.30543283, -0.02589502], [-0.00717161, 0.00318087],
                     [-0.09322739, 1.28177189], [-0.77740357, 0.74097941],
                     [-0.43877303, 1.07366684], [-0.85795321, 0.82980738],
                     [-0.30126957, -0.66268378], [0.20246714, -0.34727125]])
    y_gt = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_oss_fit_sample_with_indices():
    """Test the fit sample routine with indices support"""

    # Resample the data
    oss = OneSidedSelection(return_indices=True, random_state=RND_SEED)
    X_resampled, y_resampled, idx_under = oss.fit_sample(X, Y)

    X_gt = np.array([[-0.3879569, 0.6894251], [0.91542919, -0.65453327],
                     [-0.65571327, 0.42412021], [1.06446472, -1.09279772],
                     [0.30543283, -0.02589502], [-0.00717161, 0.00318087],
                     [-0.09322739, 1.28177189], [-0.77740357, 0.74097941],
                     [-0.43877303, 1.07366684], [-0.85795321, 0.82980738],
                     [-0.30126957, -0.66268378], [0.20246714, -0.34727125]])
    y_gt = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    idx_gt = np.array([0, 3, 9, 12, 13, 14, 1, 2, 5, 6, 8, 11])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


def test_oss_sample_wrong_X():
    """Test either if an error is raised when X is different at fitting
    and sampling"""

    # Create the object
    oss = OneSidedSelection(random_state=RND_SEED)
    oss.fit(X, Y)
    assert_raises(RuntimeError, oss.sample,
                  np.random.random((100, 40)), np.array([0] * 50 + [1] * 50))


def test_multiclass_error():
    """ Test either if an error is raised when the target are not binary
    type. """

    # continuous case
    y = np.linspace(0, 1, 15)
    oss = OneSidedSelection(random_state=RND_SEED)
    assert_warns(UserWarning, oss.fit, X, y)

    # multiclass case
    y = np.array([0] * 10 + [1] * 3 + [2] * 2)
    oss = OneSidedSelection(random_state=RND_SEED)
    assert_warns(UserWarning, oss.fit, X, y)


def test_oss_with_object():
    """Test the fit sample routine with an knn object"""

    # Resample the data
    knn = KNeighborsClassifier(n_neighbors=1)
    oss = OneSidedSelection(random_state=RND_SEED, n_neighbors=knn)
    X_resampled, y_resampled = oss.fit_sample(X, Y)

    X_gt = np.array([[-0.3879569, 0.6894251], [0.91542919, -0.65453327],
                     [-0.65571327, 0.42412021], [1.06446472, -1.09279772],
                     [0.30543283, -0.02589502], [-0.00717161, 0.00318087],
                     [-0.09322739, 1.28177189], [-0.77740357, 0.74097941],
                     [-0.43877303, 1.07366684], [-0.85795321, 0.82980738],
                     [-0.30126957, -0.66268378], [0.20246714, -0.34727125]])
    y_gt = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    # Resample the data
    knn = 1
    oss = OneSidedSelection(random_state=RND_SEED, n_neighbors=knn)
    X_resampled, y_resampled = oss.fit_sample(X, Y)
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_oss_with_wrong_object():
    """Test if an error is raised while passing a wrong object"""

    # Resample the data
    knn = 'rnd'
    oss = OneSidedSelection(random_state=RND_SEED, n_neighbors=knn)
    assert_raises(ValueError, oss.fit_sample, X, Y)
