"""Test the module Tomek's links."""
from __future__ import print_function

import numpy as np
from numpy.testing import (assert_array_equal, assert_equal, assert_raises,
                           assert_warns)
from sklearn.utils.estimator_checks import check_estimator

from imblearn.under_sampling import TomekLinks

# Generate a global dataset to use
RND_SEED = 0
X = np.array([[0.31230513, 0.1216318], [0.68481731, 0.51935141],
              [1.34192108, -0.13367336], [0.62366841, -0.21312976],
              [1.61091956, -0.40283504], [-0.37162401, -2.19400981],
              [0.74680821, 1.63827342], [0.2184254, 0.24299982],
              [0.61472253, -0.82309052], [0.19893132, -0.47761769],
              [1.06514042, -0.0770537], [0.97407872, 0.44454207],
              [1.40301027, -0.83648734], [-1.20515198, -1.02689695],
              [-0.27410027, -0.54194484], [0.8381014, 0.44085498],
              [-0.23374509, 0.18370049], [-0.32635887, -0.29299653],
              [-0.00288378, 0.84259929], [1.79580611, -0.02219234]])
Y = np.array([1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])


def test_tl_sk_estimator():
    """Test the sklearn estimator compatibility"""
    check_estimator(TomekLinks)


def test_tl_init():
    """Test the initialisation of the object"""

    # Define a ratio
    tl = TomekLinks(random_state=RND_SEED)

    assert_equal(tl.n_jobs, 1)
    assert_equal(tl.random_state, RND_SEED)


def test_tl_fit_single_class():
    """Test either if an error when there is a single class"""

    # Create the object
    tl = TomekLinks(random_state=RND_SEED)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_warns(UserWarning, tl.fit, X, y_single_class)


def test_tl_fit():
    """Test the fitting method"""

    # Create the object
    tl = TomekLinks(random_state=RND_SEED)
    # Fit the data
    tl.fit(X, Y)

    # Check if the data information have been computed
    assert_equal(tl.min_c_, 0)
    assert_equal(tl.maj_c_, 1)
    assert_equal(tl.stats_c_[0], 7)
    assert_equal(tl.stats_c_[1], 13)


def test_tl_sample_wt_fit():
    """Test either if an error is raised when sample is called before
    fitting"""

    # Create the object
    tl = TomekLinks(random_state=RND_SEED)
    assert_raises(RuntimeError, tl.sample, X, Y)


def test_tl_fit_sample():
    """Test the fit sample routine"""

    # Resample the data
    tl = TomekLinks(random_state=RND_SEED)
    X_resampled, y_resampled = tl.fit_sample(X, Y)

    X_gt = np.array([[0.31230513, 0.1216318], [0.68481731, 0.51935141],
                     [1.34192108, -0.13367336], [0.62366841, -0.21312976],
                     [1.61091956, -0.40283504], [-0.37162401, -2.19400981],
                     [0.74680821, 1.63827342], [0.2184254, 0.24299982],
                     [0.61472253, -0.82309052], [0.19893132, -0.47761769],
                     [0.97407872, 0.44454207], [1.40301027, -0.83648734],
                     [-1.20515198, -1.02689695], [-0.23374509, 0.18370049],
                     [-0.32635887, -0.29299653], [-0.00288378, 0.84259929],
                     [1.79580611, -0.02219234]])
    y_gt = np.array([1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_tl_fit_sample_with_indices():
    """Test the fit sample routine with indices support"""

    # Resample the data
    tl = TomekLinks(return_indices=True, random_state=RND_SEED)
    X_resampled, y_resampled, idx_under = tl.fit_sample(X, Y)

    X_gt = np.array([[0.31230513, 0.1216318], [0.68481731, 0.51935141],
                     [1.34192108, -0.13367336], [0.62366841, -0.21312976],
                     [1.61091956, -0.40283504], [-0.37162401, -2.19400981],
                     [0.74680821, 1.63827342], [0.2184254, 0.24299982],
                     [0.61472253, -0.82309052], [0.19893132, -0.47761769],
                     [0.97407872, 0.44454207], [1.40301027, -0.83648734],
                     [-1.20515198, -1.02689695], [-0.23374509, 0.18370049],
                     [-0.32635887, -0.29299653], [-0.00288378, 0.84259929],
                     [1.79580611, -0.02219234]])
    y_gt = np.array([1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0])
    idx_gt = np.array(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17, 18, 19])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


def test_tl_sample_wrong_X():
    """Test either if an error is raised when X is different at fitting
    and sampling"""

    # Create the object
    tl = TomekLinks(random_state=RND_SEED)
    tl.fit(X, Y)
    assert_raises(RuntimeError, tl.sample,
                  np.random.random((100, 40)), np.array([0] * 50 + [1] * 50))


def test_multiclass_error():
    """ Test either if an error is raised when the target are not binary
    type. """

    # continuous case
    y = np.linspace(0, 1, 20)
    tl = TomekLinks(random_state=RND_SEED)
    assert_warns(UserWarning, tl.fit, X, y)

    # multiclass case
    y = np.array([0] * 3 + [1] * 7 + [2] * 10)
    tl = TomekLinks(random_state=RND_SEED)
    assert_warns(UserWarning, tl.fit, X, y)
