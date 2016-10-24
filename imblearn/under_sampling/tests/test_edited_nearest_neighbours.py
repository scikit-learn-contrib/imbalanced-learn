"""Test the module edited nearest neighbour."""
from __future__ import print_function

import os
from collections import Counter

import numpy as np
from numpy.testing import (assert_array_equal, assert_equal, assert_raises,
                           assert_warns)
from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import check_estimator

from imblearn.under_sampling import EditedNearestNeighbours

# Generate a global dataset to use
RND_SEED = 0
X = np.array([[2.59928271, 0.93323465],
              [0.25738379, 0.95564169],
              [1.42772181, 0.526027],
              [1.92365863, 0.82718767],
              [-0.10903849, -0.12085181],
              [-0.284881, -0.62730973],
              [0.57062627, 1.19528323],
              [0.03394306, 0.03986753],
              [0.78318102, 2.59153329],
              [0.35831463, 1.33483198],
              [-0.14313184, -1.0412815],
              [0.01936241, 0.17799828],
              [-1.25020462, -0.40402054],
              [-0.09816301, -0.74662486],
              [-0.01252787, 0.34102657],
              [0.52726792, -0.38735648],
              [0.2821046, -0.07862747],
              [0.05230552, 0.09043907],
              [0.15198585, 0.12512646],
              [0.70524765, 0.39816382]])
Y = np.array([1, 2, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 1, 2, 1])

def test_enn_sk_estimator():
    """Test the sklearn estimator compatibility"""
    check_estimator(EditedNearestNeighbours)


def test_enn_init():
    """Test the initialisation of the object"""

    # Define a ratio
    enn = EditedNearestNeighbours(random_state=RND_SEED)

    assert_equal(enn.size_ngh, 3)
    assert_equal(enn.kind_sel, 'all')
    assert_equal(enn.n_jobs, -1)
    assert_equal(enn.random_state, RND_SEED)


def test_target_classes():
    """Test the consistency of the target classes"""

    # Define a target class from an unknown string
    target_classes = 'rnd'
    enn = EditedNearestNeighbours(target_classes=target_classes,
                                  random_state=RND_SEED)
    assert_raises(ValueError, enn.fit, X, Y)

    # Define a target class from all the possible known string
    target_classes = ('minority', 'majority', 'all', 'not minority')
    class_cases = ((0, ), (2, ), (0, 1, 2), (1, 2))
    for tc, cc in zip(target_classes, class_cases):
        enn = EditedNearestNeighbours(target_classes=tc,
                                      random_state=RND_SEED)
        enn.fit(X, Y)
        assert_equal(enn.target_classes_, cc)

    # Define a tuple which does not have the expected class inside
    target_classes = (10, 1, 2)
    enn = EditedNearestNeighbours(target_classes=target_classes,
                                  random_state=RND_SEED)
    assert_raises(ValueError, enn.fit, X, Y)

    # Check that everything goes fine while passing a right tuple
    target_classes = (1, 2)
    enn = EditedNearestNeighbours(target_classes=target_classes,
                                  random_state=RND_SEED)
    enn.fit(X, Y)
    assert_equal(target_classes, enn.target_classes_)

    # Define an unknown type for target classes
    target_classes = 5.
    enn = EditedNearestNeighbours(target_classes=target_classes,
                                  random_state=RND_SEED)
    assert_raises(ValueError, enn.fit, X, Y)


def test_enn_fit_single_class():
    """Test either if an error when there is a single class"""

    # Create the object
    enn = EditedNearestNeighbours(random_state=RND_SEED)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_warns(UserWarning, enn.fit, X, y_single_class)


def test_enn_fit():
    """Test the fitting method"""

    # Create the object
    enn = EditedNearestNeighbours(random_state=RND_SEED)
    # Fit the data
    enn.fit(X, Y)

    # Check if the data information have been computed
    assert_equal(enn.min_c_, 0)
    assert_equal(enn.maj_c_, 2)
    assert_equal(enn.stats_c_[0], 2)
    assert_equal(enn.stats_c_[1], 6)
    assert_equal(enn.stats_c_[2], 12)


def test_enn_sample_wt_fit():
    """Test either if an error is raised when sample is called before
    fitting"""

    # Create the object
    enn = EditedNearestNeighbours(random_state=RND_SEED)
    assert_raises(RuntimeError, enn.sample, X, Y)


def test_enn_fit_sample():
    """Test the fit sample routine"""

    # Resample the data
    enn = EditedNearestNeighbours(random_state=RND_SEED)
    X_resampled, y_resampled = enn.fit_sample(X, Y)

    X_gt = np.array([[-0.10903849, -0.12085181],
                  [0.01936241, 0.17799828],
                  [2.59928271, 0.93323465],
                  [1.92365863, 0.82718767],
                  [0.25738379, 0.95564169],
                  [0.78318102, 2.59153329],
                  [0.52726792, -0.38735648]])
    y_gt = np.array([0, 0, 1, 1, 2, 2, 2])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_enn_fit_sample_with_indices():
    """Test the fit sample routine with indices support"""

    # Resample the data
    enn = EditedNearestNeighbours(return_indices=True, random_state=RND_SEED)
    X_resampled, y_resampled, idx_under = enn.fit_sample(X, Y)

    X_gt = np.array([[-0.10903849, -0.12085181],
                  [0.01936241, 0.17799828],
                  [2.59928271, 0.93323465],
                  [1.92365863, 0.82718767],
                  [0.25738379, 0.95564169],
                  [0.78318102, 2.59153329],
                  [0.52726792, -0.38735648]])
    y_gt = np.array([0, 0, 1, 1, 2, 2, 2])
    idx_gt = np.array([4, 11, 0, 3, 1, 8, 15])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


def test_enn_fit_sample_mode():
    """Test the fit sample routine using the mode as selection"""

    # Resample the data
    enn = EditedNearestNeighbours(random_state=RND_SEED, kind_sel='mode')
    X_resampled, y_resampled = enn.fit_sample(X, Y)

    X_gt = np.array([[-0.10903849, -0.12085181],
                     [0.01936241, 0.17799828],
                     [2.59928271, 0.93323465],
                     [1.42772181, 0.526027],
                     [1.92365863, 0.82718767],
                     [0.25738379, 0.95564169],
                     [-0.284881, -0.62730973],
                     [0.57062627, 1.19528323],
                     [0.78318102, 2.59153329],
                     [0.35831463, 1.33483198],
                     [-0.14313184, -1.0412815],
                     [-0.09816301, -0.74662486],
                     [0.52726792, -0.38735648],
                     [0.2821046, -0.07862747]])
    y_gt = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_enn_sample_wrong_X():
    """Test either if an error is raised when X is different at fitting
    and sampling"""

    # Create the object
    enn = EditedNearestNeighbours(random_state=RND_SEED)
    enn.fit(X, Y)
    assert_raises(RuntimeError, enn.sample, np.random.random((100, 40)),
                  np.array([0] * 50 + [1] * 50))


def test_continuous_error():
    """Test either if an error is raised when the target are continuous
    type"""

    # continuous case
    y = np.linspace(0, 1, 20)
    enn = EditedNearestNeighbours(random_state=RND_SEED)
    assert_warns(UserWarning, enn.fit, X, y)
