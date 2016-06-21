"""Test the module cluster centroids."""
from __future__ import print_function

import os

import numpy as np
from numpy.testing import assert_raises
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal

from sklearn.datasets import make_classification

from unbalanced_dataset.under_sampling import ClusterCentroids

# Generate a global dataset to use
RND_SEED = 0
X, Y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=5000, random_state=RND_SEED)


def test_cc_bad_ratio():
    """Test either if an error is raised with a wrong decimal value for
    the ratio"""

    # Define a negative ratio
    ratio = -1.0
    assert_raises(ValueError, ClusterCentroids, ratio=ratio)

    # Define a ratio greater than 1
    ratio = 100.0
    assert_raises(ValueError, ClusterCentroids, ratio=ratio)

    # Define ratio as an unknown string
    ratio = 'rnd'
    assert_raises(ValueError, ClusterCentroids, ratio=ratio)

    # Define ratio as a list which is not supported
    ratio = [.5, .5]
    assert_raises(ValueError, ClusterCentroids, ratio=ratio)


def test_init():
    """Test the initialisation of the object"""

    # Define a ratio
    ratio = 1.
    verbose = True
    cc = ClusterCentroids(ratio=ratio, random_state=RND_SEED, verbose=verbose)

    assert_equal(cc.ratio_, ratio)
    assert_equal(cc.rs_, RND_SEED)
    assert_equal(cc.verbose, verbose)
    assert_equal(cc.min_c_, None)
    assert_equal(cc.maj_c_, None)
    assert_equal(cc.stats_c_, {})


def test_cc_fit_single_class():
    """Test either if an error when there is a single class"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    cc = ClusterCentroids(ratio=ratio, random_state=RND_SEED)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_raises(RuntimeError, cc.fit, X, y_single_class)


def test_cc_fit_invalid_ratio():
    """Test either if an error is raised when the balancing ratio to fit is
    smaller than the one of the data"""

    # Create the object
    ratio = 1. / 10000.
    cc = ClusterCentroids(ratio=ratio, random_state=RND_SEED)
    # Fit the data
    assert_raises(RuntimeError, cc.fit, X, Y)


def test_cc_fit():
    """Test the fitting method"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    cc = ClusterCentroids(ratio=ratio, random_state=RND_SEED)
    # Fit the data
    cc.fit(X, Y)

    # Check if the data information have been computed
    assert_equal(cc.min_c_, 0)
    assert_equal(cc.maj_c_, 1)
    assert_equal(cc.stats_c_[0], 500)
    assert_equal(cc.stats_c_[1], 4500)


def test_transform_wt_fit():
    """Test either if an error is raised when transform is called before
    fitting"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    cc = ClusterCentroids(ratio=ratio, random_state=RND_SEED)
    assert_raises(RuntimeError, cc.transform, X, Y)


def test_fit_transform_auto():
    """Test fit and transform routines with auto ratio"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    cc = ClusterCentroids(ratio=ratio, random_state=RND_SEED)

    # Fit and transform
    X_resampled, y_resampled = cc.fit_transform(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'cc_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'cc_y.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_fit_transform_half():
    """Test fit and transform routines with ratio of .5"""

    # Define the parameter for the under-sampling
    ratio = .5

    # Create the object
    cc = ClusterCentroids(ratio=ratio, random_state=RND_SEED)

    # Fit and transform
    X_resampled, y_resampled = cc.fit_transform(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'cc_x_05.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'cc_y_05.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
