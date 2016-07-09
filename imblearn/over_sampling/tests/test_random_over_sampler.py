"""Test the module under sampler."""
from __future__ import print_function

import os

import numpy as np
from numpy.testing import assert_raises
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_warns

from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import check_estimator

from imblearn.over_sampling import RandomOverSampler

# Generate a global dataset to use
RND_SEED = 0
X, Y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=5000, random_state=RND_SEED)


def test_ros_sk_estimator():
    """Test the sklearn estimator compatibility"""
    check_estimator(RandomOverSampler)


def test_ros_bad_ratio():
    """Test either if an error is raised with a wrong decimal value for
    the ratio"""

    # Define a negative ratio
    ratio = -1.0
    ros = RandomOverSampler(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, ros.fit_sample, X, Y)

    # Define a ratio greater than 1
    ratio = 100.0
    ros = RandomOverSampler(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, ros.fit_sample, X, Y)

    # Define ratio as an unknown string
    ratio = 'rnd'
    ros = RandomOverSampler(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, ros.fit_sample, X, Y)

    # Define ratio as a list which is not supported
    ratio = [.5, .5]
    ros = RandomOverSampler(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, ros.fit_sample, X, Y)


def test_ros_init():
    """Test the initialisation of the object"""

    # Define a ratio
    ratio = 'auto'
    ros = RandomOverSampler(ratio=ratio, random_state=RND_SEED)

    assert_equal(ros.random_state, RND_SEED)


def test_ros_fit_single_class():
    """Test either if an error when there is a single class"""

    # Create the object
    ros = RandomOverSampler(random_state=RND_SEED)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_warns(RuntimeWarning, ros.fit, X, y_single_class)


def test_ros_fit_invalid_ratio():
    """Test either if an error is raised when the balancing ratio to fit is
    smaller than the one of the data"""

    # Create the object
    ratio = 1. / 10000.
    ros = RandomOverSampler(ratio=ratio, random_state=RND_SEED)
    # Fit the data
    assert_raises(RuntimeError, ros.fit, X, Y)


def test_ros_fit():
    """Test the fitting method"""

    # Create the object
    ros = RandomOverSampler(random_state=RND_SEED)
    # Fit the data
    ros.fit(X, Y)

    # Check if the data information have been computed
    assert_equal(ros.min_c_, 0)
    assert_equal(ros.maj_c_, 1)
    assert_equal(ros.stats_c_[0], 500)
    assert_equal(ros.stats_c_[1], 4500)


def test_ros_sample_wt_fit():
    """Test either if an error is raised when sample is called before
    fitting"""

    # Create the object
    ros = RandomOverSampler(random_state=RND_SEED)
    assert_raises(RuntimeError, ros.sample, X, Y)


def test_ros_fit_sample():
    """Test the fit sample routine"""

    # Resample the data
    ros = RandomOverSampler(random_state=RND_SEED)
    X_resampled, y_resampled = ros.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'ros_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'ros_y.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_ros_fit_sample_half():
    """Test the fit sample routine with a 0.5 ratio"""

    # Resample the data
    ratio = 0.5
    ros = RandomOverSampler(ratio=ratio, random_state=RND_SEED)
    X_resampled, y_resampled = ros.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'ros_x_05.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'ros_y_05.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_sample_wrong_X():
    """Test either if an error is raised when X is different at fitting
    and sampling"""

    # Create the object
    ros = RandomOverSampler(random_state=RND_SEED)
    ros.fit(X, Y)
    assert_raises(RuntimeError, ros.sample, np.random.random((100, 40)),
                  np.array([0] * 50 + [1] * 50))
