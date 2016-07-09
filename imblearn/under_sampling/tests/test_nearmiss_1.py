"""Test the module nearmiss."""
from __future__ import print_function

import os

import numpy as np
from numpy.testing import assert_raises
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_warns

from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import check_estimator

from imblearn.under_sampling import NearMiss

# Generate a global dataset to use
RND_SEED = 0
X, Y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=5000, random_state=RND_SEED)
VERSION_NEARMISS = 1


def test_nearmiss_sk_estimator():
    """Test the sklearn estimator compatibility"""
    check_estimator(NearMiss)


def test_nearmiss_bad_ratio():
    """Test either if an error is raised with a wrong decimal value for
    the ratio"""

    # Define a negative ratio
    ratio = -1.0
    nm1 = NearMiss(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, nm1.fit, X, Y)

    # Define a ratio greater than 1
    ratio = 100.0
    nm1 = NearMiss(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, nm1.fit, X, Y)

    # Define ratio as an unknown string
    ratio = 'rnd'
    nm1 = NearMiss(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, nm1.fit, X, Y)

    # Define ratio as a list which is not supported
    ratio = [.5, .5]
    nm1 = NearMiss(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, nm1.fit, X, Y)


def test_nearmiss_wrong_version():
    """Test either if an error is raised when the version is unknown."""

    version = 1000
    nm1 = NearMiss(version=version, random_state=RND_SEED)
    assert_raises(ValueError, nm1.fit_sample, X, Y)


def test_nearmiss_init():
    """Test the initialisation of the object"""

    # Define a ratio
    ratio = 1.
    nm1 = NearMiss(ratio=ratio, random_state=RND_SEED,
                   version=VERSION_NEARMISS)

    assert_equal(nm1.version, VERSION_NEARMISS)
    assert_equal(nm1.size_ngh, 3)
    assert_equal(nm1.ratio, ratio)
    assert_equal(nm1.random_state, RND_SEED)


def test_nearmiss_fit_single_class():
    """Test either if an error when there is a single class"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    nm1 = NearMiss(ratio=ratio, random_state=RND_SEED,
                   version=VERSION_NEARMISS)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_warns(RuntimeWarning, nm1.fit, X, y_single_class)


def test_nm_fit_invalid_ratio():
    """Test either if an error is raised when the balancing ratio to fit is
    smaller than the one of the data"""

    # Create the object
    ratio = 1. / 10000.
    nm = NearMiss(ratio=ratio, random_state=RND_SEED)
    # Fit the data
    assert_raises(RuntimeError, nm.fit, X, Y)


def test_nm1_fit():
    """Test the fitting method"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    nm1 = NearMiss(ratio=ratio, random_state=RND_SEED,
                   version=VERSION_NEARMISS)
    # Fit the data
    nm1.fit(X, Y)

    # Check if the data information have been computed
    assert_equal(nm1.min_c_, 0)
    assert_equal(nm1.maj_c_, 1)
    assert_equal(nm1.stats_c_[0], 500)
    assert_equal(nm1.stats_c_[1], 4500)


def test_nm1_sample_wt_fit():
    """Test either if an error is raised when sample is called before
    fitting"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    nm1 = NearMiss(ratio=ratio, random_state=RND_SEED,
                   version=VERSION_NEARMISS)
    assert_raises(RuntimeError, nm1.sample, X, Y)


def test_nm1_fit_sample_auto():
    """Test fit and sample routines with auto ratio"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    nm1 = NearMiss(ratio=ratio, random_state=RND_SEED,
                   version=VERSION_NEARMISS)

    # Fit and sample
    X_resampled, y_resampled = nm1.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'nm1_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'nm1_y.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_nm1_fit_sample_auto_indices():
    """Test fit and sample routines with auto ratio and indices support"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    nm1 = NearMiss(ratio=ratio, random_state=RND_SEED,
                   version=VERSION_NEARMISS, return_indices=True)

    # Fit and sample
    X_resampled, y_resampled, idx_under = nm1.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'nm1_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'nm1_y.npy'))
    idx_gt = np.load(os.path.join(currdir, 'data', 'nm1_idx.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


def test_nm1_fit_sample_half():
    """Test fit and sample routines with .5 ratio"""

    # Define the parameter for the under-sampling
    ratio = .5

    # Create the object
    nm1 = NearMiss(ratio=ratio, random_state=RND_SEED,
                   version=VERSION_NEARMISS)

    # Fit and sample
    X_resampled, y_resampled = nm1.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'nm1_x_05.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'nm1_y_05.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_nm1_sample_wrong_X():
    """Test either if an error is raised when X is different at fitting
    and sampling"""

    # Create the object
    nm1 = NearMiss(random_state=RND_SEED)
    nm1.fit(X, Y)
    assert_raises(RuntimeError, nm1.sample, np.random.random((100, 40)),
                  np.array([0] * 50 + [1] * 50))
