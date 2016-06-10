"""Test the module Tomek's links."""
from __future__ import print_function

import os

import numpy as np
from numpy.testing import assert_raises
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal

from sklearn.datasets import make_classification

from unbalanced_dataset.under_sampling import TomekLinks

# Generate a global dataset to use
RND_SEED = 0
X, Y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=5000, random_state=RND_SEED)


def test_tl_init():
    """Test the initialisation of the object"""

    # Define a ratio
    verbose = True
    tl = TomekLinks(random_state=RND_SEED, verbose=verbose)

    assert_equal(tl.n_jobs, -1)
    assert_equal(tl.rs_, RND_SEED)
    assert_equal(tl.verbose, verbose)
    assert_equal(tl.min_c_, None)
    assert_equal(tl.maj_c_, None)
    assert_equal(tl.stats_c_, {})


def test_tl_fit_single_class():
    """Test either if an error when there is a single class"""

    # Create the object
    tl = TomekLinks(random_state=RND_SEED)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_raises(RuntimeError, tl.fit, X, y_single_class)


def test_tl_fit():
    """Test the fitting method"""

    # Create the object
    tl = TomekLinks(random_state=RND_SEED)
    # Fit the data
    tl.fit(X, Y)

    # Check if the data information have been computed
    assert_equal(tl.min_c_, 0)
    assert_equal(tl.maj_c_, 1)
    assert_equal(tl.stats_c_[0], 500)
    assert_equal(tl.stats_c_[1], 4500)


def test_tl_transform_wt_fit():
    """Test either if an error is raised when transform is called before
    fitting"""

    # Create the object
    tl = TomekLinks(random_state=RND_SEED)
    assert_raises(RuntimeError, tl.transform, X, Y)


def test_tl_fit_transform():
    """Test the fit transform routine"""

    # Resample the data
    tl = TomekLinks(random_state=RND_SEED)
    X_resampled, y_resampled = tl.fit_transform(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'tl_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'tl_y.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_tl_fit_transform_with_indices():
    """Test the fit transform routine with indices support"""

    # Resample the data
    tl = TomekLinks(return_indices=True, random_state=RND_SEED)
    X_resampled, y_resampled, idx_under = tl.fit_transform(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'tl_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'tl_y.npy'))
    idx_gt = np.load(os.path.join(currdir, 'data', 'tl_idx.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)
