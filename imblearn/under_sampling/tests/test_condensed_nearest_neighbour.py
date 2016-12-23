"""Test the module condensed nearest neighbour."""
from __future__ import print_function

import numpy as np
from numpy.testing import (assert_array_equal, assert_equal, assert_raises,
                           assert_warns)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.estimator_checks import check_estimator

from imblearn.under_sampling import CondensedNearestNeighbour

# Generate a global dataset to use
RND_SEED = 0
X = np.array([[2.59928271, 0.93323465], [0.25738379, 0.95564169],
              [1.42772181, 0.526027], [1.92365863, 0.82718767],
              [-0.10903849, -0.12085181], [-0.284881, -0.62730973],
              [0.57062627, 1.19528323], [0.03394306, 0.03986753],
              [0.78318102, 2.59153329], [0.35831463, 1.33483198],
              [-0.14313184, -1.0412815], [0.01936241, 0.17799828],
              [-1.25020462, -0.40402054], [-0.09816301, -0.74662486],
              [-0.01252787, 0.34102657], [0.52726792, -0.38735648],
              [0.2821046, -0.07862747], [0.05230552, 0.09043907],
              [0.15198585, 0.12512646], [0.70524765, 0.39816382]])
Y = np.array([1, 2, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 1, 2, 1])


def test_cnn_sk_estimator():
    """Test the sklearn estimator compatibility"""
    check_estimator(CondensedNearestNeighbour)


def test_cnn_init():
    """Test the initialisation of the object"""

    # Define a ratio
    cnn = CondensedNearestNeighbour(random_state=RND_SEED)

    assert_equal(cnn.n_seeds_S, 1)
    assert_equal(cnn.n_jobs, 1)


def test_cnn_fit_single_class():
    """Test either if an error when there is a single class"""

    # Create the object
    cnn = CondensedNearestNeighbour(random_state=RND_SEED)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_warns(UserWarning, cnn.fit, X, y_single_class)


def test_cnn_fit():
    """Test the fitting method"""

    # Create the object
    cnn = CondensedNearestNeighbour(random_state=RND_SEED)
    # Fit the data
    cnn.fit(X, Y)

    # Check if the data information have been computed
    assert_equal(cnn.min_c_, 0)
    assert_equal(cnn.maj_c_, 2)
    assert_equal(cnn.stats_c_[0], 2)
    assert_equal(cnn.stats_c_[1], 6)
    assert_equal(cnn.stats_c_[2], 12)


def test_cnn_sample_wt_fit():
    """Test either if an error is raised when sample is called before
    fitting"""

    # Create the object
    cnn = CondensedNearestNeighbour(random_state=RND_SEED)
    assert_raises(RuntimeError, cnn.sample, X, Y)


def test_cnn_fit_sample():
    """Test the fit sample routine"""

    # Resample the data
    cnn = CondensedNearestNeighbour(random_state=RND_SEED)
    X_resampled, y_resampled = cnn.fit_sample(X, Y)

    X_gt = np.array([[-0.10903849, -0.12085181], [0.01936241, 0.17799828],
                     [0.05230552, 0.09043907], [-1.25020462, -0.40402054],
                     [0.70524765, 0.39816382], [0.35831463, 1.33483198],
                     [-0.284881, -0.62730973], [0.03394306, 0.03986753],
                     [-0.01252787, 0.34102657], [0.15198585, 0.12512646]])
    y_gt = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_cnn_fit_sample_with_indices():
    """Test the fit sample routine with indices support"""

    # Resample the data
    cnn = CondensedNearestNeighbour(return_indices=True, random_state=RND_SEED)
    X_resampled, y_resampled, idx_under = cnn.fit_sample(X, Y)

    X_gt = np.array([[-0.10903849, -0.12085181], [0.01936241, 0.17799828],
                     [0.05230552, 0.09043907], [-1.25020462, -0.40402054],
                     [0.70524765, 0.39816382], [0.35831463, 1.33483198],
                     [-0.284881, -0.62730973], [0.03394306, 0.03986753],
                     [-0.01252787, 0.34102657], [0.15198585, 0.12512646]])
    y_gt = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    idx_gt = np.array([4, 11, 17, 12, 19, 9, 5, 7, 14, 18])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


def test_cnn_sample_wrong_X():
    """Test either if an error is raised when X is different at fitting
    and sampling"""

    # Create the object
    cnn = CondensedNearestNeighbour(random_state=RND_SEED)
    cnn.fit(X, Y)
    assert_raises(RuntimeError, cnn.sample,
                  np.random.random((100, 40)), np.array([0] * 50 + [1] * 50))


def test_continuous_error():
    """Test either if an error is raised when the target are continuous
    type"""

    # continuous case
    y = np.linspace(0, 1, 20)
    cnn = CondensedNearestNeighbour(random_state=RND_SEED)
    assert_warns(UserWarning, cnn.fit, X, y)


def test_cnn_fit_sample_with_object():
    """Test the fit sample routine with a knn object"""

    # Resample the data
    knn = KNeighborsClassifier(n_neighbors=1)
    cnn = CondensedNearestNeighbour(random_state=RND_SEED, n_neighbors=knn)
    X_resampled, y_resampled = cnn.fit_sample(X, Y)

    X_gt = np.array([[-0.10903849, -0.12085181], [0.01936241, 0.17799828],
                     [0.05230552, 0.09043907], [-1.25020462, -0.40402054],
                     [0.70524765, 0.39816382], [0.35831463, 1.33483198],
                     [-0.284881, -0.62730973], [0.03394306, 0.03986753],
                     [-0.01252787, 0.34102657], [0.15198585, 0.12512646]])
    y_gt = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)

    cnn = CondensedNearestNeighbour(random_state=RND_SEED, n_neighbors=1)
    X_resampled, y_resampled = cnn.fit_sample(X, Y)
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_cnn_fit_sample_with_wrong_object():
    """Test either if an error is raised while a wrong object is given"""

    # Resample the data
    knn = 'rnd'
    cnn = CondensedNearestNeighbour(random_state=RND_SEED, n_neighbors=knn)
    assert_raises(ValueError, cnn.fit_sample, X, Y)
