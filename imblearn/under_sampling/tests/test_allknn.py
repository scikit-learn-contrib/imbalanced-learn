"""Test the module repeated edited nearest neighbour."""
from __future__ import print_function

import os

import numpy as np
from numpy.testing import assert_raises
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_warns

from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import check_estimator

from imblearn.under_sampling import AllKNN

# Generate a global dataset to use
RND_SEED = 0
X, Y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=5000, random_state=RND_SEED)


def test_allknn_sk_estimator():
    """Test the sklearn estimator compatibility"""
    check_estimator(AllKNN)


def test_allknn_init():
    """Test the initialisation of the object"""

    # Define a ratio
    allknn = AllKNN(random_state=RND_SEED)

    assert_equal(allknn.size_ngh, 3)
    assert_equal(allknn.kind_sel, 'all')
    assert_equal(allknn.n_jobs, -1)
    assert_equal(allknn.random_state, RND_SEED)


def test_allknn_fit_single_class():
    """Test either if an error when there is a single class"""

    # Create the object
    allknn = AllKNN(random_state=RND_SEED)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_warns(RuntimeWarning, allknn.fit, X, y_single_class)


def test_allknn_fit():
    """Test the fitting method"""

    # Create the object
    allknn = AllKNN(random_state=RND_SEED)
    # Fit the data
    allknn.fit(X, Y)

    # Check if the data information have been computed
    assert_equal(allknn.min_c_, 0)
    assert_equal(allknn.maj_c_, 1)
    assert_equal(allknn.stats_c_[0], 500)
    assert_equal(allknn.stats_c_[1], 4500)


def test_allknn_sample_wt_fit():
    """Test either if an error is raised when sample is called before
    fitting"""

    # Create the object
    allknn = AllKNN(random_state=RND_SEED)
    assert_raises(RuntimeError, allknn.sample, X, Y)


def test_allknn_fit_sample():
    """Test the fit sample routine"""

    # Resample the data
    allknn = AllKNN(random_state=RND_SEED)
    X_resampled, y_resampled = allknn.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'allknn_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'allknn_y.npy'))
    assert_array_almost_equal(X_resampled, X_gt)
    assert_array_almost_equal(y_resampled, y_gt)


def test_allknn_fit_sample_with_indices():
    """Test the fit sample routine with indices support"""

    # Resample the data
    allknn = AllKNN(return_indices=True, random_state=RND_SEED)
    X_resampled, y_resampled, idx_under = allknn.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'allknn_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'allknn_y.npy'))
    idx_gt = np.load(os.path.join(currdir, 'data', 'allknn_idx.npy'))
    assert_array_almost_equal(X_resampled, X_gt)
    assert_array_almost_equal(y_resampled, y_gt)
    assert_array_almost_equal(idx_under, idx_gt)


def test_allknn_fit_sample_mode():
    """Test the fit sample routine using the mode as selection"""

    # Resample the data
    allknn = AllKNN(random_state=RND_SEED, kind_sel='mode')
    X_resampled, y_resampled = allknn.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'allknn_x_mode.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'allknn_y_mode.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_almost_equal(y_resampled, y_gt)


def test_allknn_sample_wrong_X():
    """Test either if an error is raised when X is different at fitting
    and sampling"""

    # Create the object
    allknn = AllKNN(random_state=RND_SEED)
    allknn.fit(X, Y)
    assert_raises(RuntimeError, allknn.sample, np.random.random((100, 40)),
                  np.array([0] * 50 + [1] * 50))
