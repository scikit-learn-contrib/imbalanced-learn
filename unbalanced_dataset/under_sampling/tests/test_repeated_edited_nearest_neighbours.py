"""Test the module repeated edited nearest neighbour."""
from __future__ import print_function

import os

import numpy as np
from numpy.testing import assert_raises
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_warns

from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import check_estimator

from unbalanced_dataset.under_sampling import RepeatedEditedNearestNeighbours

# Generate a global dataset to use
RND_SEED = 0
X, Y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=5000, random_state=RND_SEED)


def test_renn_init():
    """Test the initialisation of the object"""

    # Define a ratio
    verbose = True
    renn = RepeatedEditedNearestNeighbours(random_state=RND_SEED,
                                           verbose=verbose)

    assert_equal(renn.size_ngh, 3)
    assert_equal(renn.kind_sel, 'all')
    assert_equal(renn.n_jobs, -1)
    assert_equal(renn.rs_, RND_SEED)
    assert_equal(renn.verbose, verbose)
    assert_equal(renn.min_c_, None)
    assert_equal(renn.maj_c_, None)
    assert_equal(renn.stats_c_, {})


def test_renn_fit_single_class():
    """Test either if an error when there is a single class"""

    # Create the object
    renn = RepeatedEditedNearestNeighbours(random_state=RND_SEED)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_raises(RuntimeError, renn.fit, X, y_single_class)


def test_renn_fit():
    """Test the fitting method"""

    # Create the object
    renn = RepeatedEditedNearestNeighbours(random_state=RND_SEED)
    # Fit the data
    renn.fit(X, Y)

    # Check if the data information have been computed
    assert_equal(renn.min_c_, 0)
    assert_equal(renn.maj_c_, 1)
    assert_equal(renn.stats_c_[0], 500)
    assert_equal(renn.stats_c_[1], 4500)


def test_renn_transform_wt_fit():
    """Test either if an error is raised when transform is called before
    fitting"""

    # Create the object
    renn = RepeatedEditedNearestNeighbours(random_state=RND_SEED)
    assert_raises(RuntimeError, renn.transform, X, Y)


def test_renn_fit_transform():
    """Test the fit transform routine"""

    # Resample the data
    renn = RepeatedEditedNearestNeighbours(random_state=RND_SEED)
    X_resampled, y_resampled = renn.fit_transform(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'renn_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'renn_y.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_renn_fit_transform_with_indices():
    """Test the fit transform routine with indices support"""

    # Resample the data
    renn = RepeatedEditedNearestNeighbours(return_indices=True,
                                           random_state=RND_SEED)
    X_resampled, y_resampled, idx_under = renn.fit_transform(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'renn_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'renn_y.npy'))
    idx_gt = np.load(os.path.join(currdir, 'data', 'renn_idx.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


def test_renn_fit_transform_mode():
    """Test the fit transform routine using the mode as selection"""

    # Resample the data
    renn = RepeatedEditedNearestNeighbours(random_state=RND_SEED,
                                           kind_sel='mode')
    X_resampled, y_resampled = renn.fit_transform(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'renn_x_mode.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'renn_y_mode.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
