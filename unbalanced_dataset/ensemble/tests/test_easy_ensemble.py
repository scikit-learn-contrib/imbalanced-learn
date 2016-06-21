"""Test the module easy ensemble."""
from __future__ import print_function

import os

import numpy as np
from numpy.testing import assert_raises
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal

from sklearn.datasets import make_classification

from unbalanced_dataset.ensemble import EasyEnsemble

# Generate a global dataset to use
RND_SEED = 0
X, Y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=5000, random_state=RND_SEED)


def test_ee_bad_ratio():
    """Test either if an error is raised with a wrong decimal value for
    the ratio"""

    # Define a negative ratio
    ratio = -1.0
    assert_raises(ValueError, EasyEnsemble, ratio=ratio)

    # Define a ratio greater than 1
    ratio = 100.0
    assert_raises(ValueError, EasyEnsemble, ratio=ratio)

    # Define ratio as an unknown string
    ratio = 'rnd'
    assert_raises(ValueError, EasyEnsemble, ratio=ratio)

    # Define ratio as a list which is not supported
    ratio = [.5, .5]
    assert_raises(ValueError, EasyEnsemble, ratio=ratio)


def test_ee_init():
    """Test the initialisation of the object"""

    # Define a ratio
    ratio = 1.
    verbose = True
    ee = EasyEnsemble(ratio=ratio, random_state=RND_SEED, verbose=verbose)

    assert_equal(ee.ratio_, ratio)
    assert_equal(ee.replacement, False)
    assert_equal(ee.n_subsets, 10)
    assert_equal(ee.rs_, RND_SEED)
    assert_equal(ee.verbose, verbose)
    assert_equal(ee.min_c_, None)
    assert_equal(ee.maj_c_, None)
    assert_equal(ee.stats_c_, {})


def test_ee_fit_single_class():
    """Test either if an error when there is a single class"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    ee = EasyEnsemble(ratio=ratio, random_state=RND_SEED)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_raises(RuntimeError, ee.fit, X, y_single_class)


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
    assert_equal(ee.stats_c_[0], 500)
    assert_equal(ee.stats_c_[1], 4500)


def test_transform_wt_fit():
    """Test either if an error is raised when transform is called before
    fitting"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    ee = EasyEnsemble(ratio=ratio, random_state=RND_SEED)
    assert_raises(RuntimeError, ee.transform, X, Y)


def test_fit_transform_auto():
    """Test the fit and transform routine with auto ratio."""

    # Define the ratio parameter
    ratio = 'auto'

    # Create the sampling object
    ee = EasyEnsemble(ratio=ratio, random_state=RND_SEED,
                      return_indices=True)

    # Get the different subset
    X_resampled, y_resampled, idx_under = ee.fit_transform(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'ee_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'ee_y.npy'))
    idx_gt = np.load(os.path.join(currdir, 'data', 'ee_idx.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


def test_fit_transform_half():
    """Test the fit and transform routine with 0.5 ratio."""

    # Define the ratio parameter
    ratio = 0.5

    # Create the sampling object
    ee = EasyEnsemble(ratio=ratio, random_state=RND_SEED)

    # Get the different subset
    X_resampled, y_resampled = ee.fit_transform(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'ee_x_05.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'ee_y_05.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
