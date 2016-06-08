"""Test the module nearmiss."""
from __future__ import print_function

import os

import numpy as np
from numpy.testing import assert_raises
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal

from sklearn.datasets import make_classification

from unbalanced_dataset.under_sampling import NearMiss

# Generate a global dataset to use
RND_SEED = 0
X, Y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=5000, random_state=RND_SEED)
VERSION_NEARMISS = 2


def test_nearmiss_bad_ratio():
    """Test either if an error is raised with a wrong decimal value for
    the ratio"""

    # Define a negative ratio
    ratio = -1.0
    assert_raises(ValueError, NearMiss, ratio=ratio)

    # Define a ratio greater than 1
    ratio = 100.0
    assert_raises(ValueError, NearMiss, ratio=ratio)

    # Define ratio as an unknown string
    ratio = 'rnd'
    assert_raises(ValueError, NearMiss, ratio=ratio)

    # Define ratio as a list which is not supported
    ratio = [.5, .5]
    assert_raises(ValueError, NearMiss, ratio=ratio)


def test_nearmiss_wrong_version():
    """Test either if an error is raised when the version is unknown."""

    version = 1000
    assert_raises(ValueError, NearMiss, version=version)


def test_nearmiss_init():
    """Test the initialisation of the object"""

    # Define a ratio
    ratio = 1.
    verbose = True
    nm2 = NearMiss(ratio=ratio, random_state=RND_SEED, verbose=verbose,
                   version=VERSION_NEARMISS)

    assert_equal(nm2.version, VERSION_NEARMISS)
    assert_equal(nm2.size_ngh, 3)
    assert_equal(nm2.ratio_, ratio)
    assert_equal(nm2.rs_, RND_SEED)
    assert_equal(nm2.verbose, verbose)
    assert_equal(nm2.min_c_, None)
    assert_equal(nm2.maj_c_, None)
    assert_equal(nm2.stats_c_, {})


def test_nearmiss_fit_single_class():
    """Test either if an error when there is a single class"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    nm2 = NearMiss(ratio=ratio, random_state=RND_SEED,
                   version=VERSION_NEARMISS)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_raises(RuntimeError, nm2.fit, X, y_single_class)


def test_nm2_fit():
    """Test the fitting method"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    nm2 = NearMiss(ratio=ratio, random_state=RND_SEED,
                   version=VERSION_NEARMISS)
    # Fit the data
    nm2.fit(X, Y)

    # Check if the data information have been computed
    assert_equal(nm2.min_c_, 0)
    assert_equal(nm2.maj_c_, 1)
    assert_equal(nm2.stats_c_[0], 500)
    assert_equal(nm2.stats_c_[1], 4500)


def test_nm2_transform_wt_fit():
    """Test either if an error is raised when transform is called before
    fitting"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    nm2 = NearMiss(ratio=ratio, random_state=RND_SEED,
                          version=VERSION_NEARMISS)
    assert_raises(RuntimeError, nm2.transform, X, Y)


def test_nm2_fit_transform_auto():
    """Test fit and transform routines with auto ratio"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    nm2 = NearMiss(ratio=ratio, random_state=RND_SEED,
                   version=VERSION_NEARMISS)

    # Fit and transform
    X_resampled, y_resampled = nm2.fit_transform(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'nm2_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'nm2_y.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_nm2_fit_transform_auto_indices():
    """Test fit and transform routines with auto ratio and indices support"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    nm2 = NearMiss(ratio=ratio, random_state=RND_SEED,
                   version=VERSION_NEARMISS, return_indices=True)

    # Fit and transform
    X_resampled, y_resampled, idx_under = nm2.fit_transform(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'nm2_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'nm2_y.npy'))
    idx_gt = np.load(os.path.join(currdir, 'data', 'nm2_idx.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


def test_nm2_fit_transform_half():
    """Test fit and transform routines with .5 ratio"""

    # Define the parameter for the under-sampling
    ratio = .5

    # Create the object
    nm2 = NearMiss(ratio=ratio, random_state=RND_SEED,
                   version=VERSION_NEARMISS)

    # Fit and transform
    X_resampled, y_resampled = nm2.fit_transform(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'nm2_x_05.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'nm2_y_05.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
