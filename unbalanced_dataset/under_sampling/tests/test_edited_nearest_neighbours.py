"""Test the module condensed nearest neighbour."""
from __future__ import print_function

import os

import numpy as np
from numpy.testing import assert_raises
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal

from sklearn.datasets import make_classification

from unbalanced_dataset.under_sampling import EditedNearestNeighbours

# Generate a global dataset to use
RND_SEED = 0
X, Y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=5000, random_state=RND_SEED)


def test_enn_init():
    """Test the initialisation of the object"""

    # Define a ratio
    verbose = True
    enn = EditedNearestNeighbours(random_state=RND_SEED, verbose=verbose)

    assert_equal(enn.size_ngh, 3)
    assert_equal(enn.kind_sel, 'all')
    assert_equal(enn.n_jobs, -1)
    assert_equal(enn.rs_, RND_SEED)
    assert_equal(enn.verbose, verbose)
    assert_equal(enn.min_c_, None)
    assert_equal(enn.maj_c_, None)
    assert_equal(enn.stats_c_, {})


def test_enn_fit_single_class():
    """Test either if an error when there is a single class"""

    # Create the object
    enn = EditedNearestNeighbours(random_state=RND_SEED)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_raises(RuntimeError, enn.fit, X, y_single_class)


def test_enn_fit():
    """Test the fitting method"""

    # Create the object
    enn = EditedNearestNeighbours(random_state=RND_SEED)
    # Fit the data
    enn.fit(X, Y)

    # Check if the data information have been computed
    assert_equal(enn.min_c_, 0)
    assert_equal(enn.maj_c_, 1)
    assert_equal(enn.stats_c_[0], 500)
    assert_equal(enn.stats_c_[1], 4500)


def test_enn_transform_wt_fit():
    """Test either if an error is raised when transform is called before
    fitting"""

    # Create the object
    enn = EditedNearestNeighbours(random_state=RND_SEED)
    assert_raises(RuntimeError, enn.transform, X, Y)


def test_enn_fit_transform():
    """Test the fit transform routine"""

    # Resample the data
    enn = EditedNearestNeighbours(random_state=RND_SEED)
    X_resampled, y_resampled = enn.fit_transform(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'enn_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'enn_y.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_enn_fit_transform_with_indices():
    """Test the fit transform routine with indices support"""

    # Resample the data
    enn = EditedNearestNeighbours(return_indices=True, random_state=RND_SEED)
    X_resampled, y_resampled, idx_under = enn.fit_transform(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'enn_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'enn_y.npy'))
    idx_gt = np.load(os.path.join(currdir, 'data', 'enn_idx.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


def test_enn_fit_transform_mode():
    """Test the fit transform routine using the mode as selection"""

    # Resample the data
    enn = EditedNearestNeighbours(random_state=RND_SEED, kind_sel='mode')
    X_resampled, y_resampled = enn.fit_transform(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    np.save(os.path.join(currdir, 'data', 'enn_x_mode.npy'), X_resampled)
    np.save(os.path.join(currdir, 'data', 'enn_y_mode.npy'), y_resampled)
    X_gt = np.load(os.path.join(currdir, 'data', 'enn_x_mode.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'enn_y_mode.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
