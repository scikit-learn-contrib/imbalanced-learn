"""Test the module cluster centroids."""
from __future__ import print_function

from collections import Counter

import numpy as np
from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_equal, assert_raises, assert_warns)
from sklearn.cluster import KMeans
from sklearn.utils.estimator_checks import check_estimator

from imblearn.under_sampling import ClusterCentroids

# Generate a global dataset to use
RND_SEED = 0
# Data generated for the toy example
X = np.array([[0.04352327, -0.20515826], [0.92923648, 0.76103773],
              [0.20792588, 1.49407907], [0.47104475, 0.44386323],
              [0.22950086, 0.33367433], [0.15490546, 0.3130677],
              [0.09125309, -0.85409574], [0.12372842, 0.6536186],
              [0.13347175, 0.12167502], [0.094035, -2.55298982]])
Y = np.array([1, 0, 1, 0, 1, 1, 1, 1, 0, 1])


def test_cc_sk_estimator():
    """Test the sklearn estimator compatibility"""
    check_estimator(ClusterCentroids)


def test_cc_bad_ratio():
    """Test either if an error is raised with a wrong decimal value for
    the ratio"""

    # Define a negative ratio
    ratio = -1.0
    cc = ClusterCentroids(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, cc.fit, X, Y)

    # Define a ratio greater than 1
    ratio = 100.0
    cc = ClusterCentroids(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, cc.fit, X, Y)

    # Define ratio as an unknown string
    ratio = 'rnd'
    cc = ClusterCentroids(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, cc.fit, X, Y)

    # Define ratio as a list which is not supported
    ratio = [.5, .5]
    cc = ClusterCentroids(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, cc.fit, X, Y)


def test_init():
    """Test the initialisation of the object"""

    # Define a ratio
    ratio = 1.
    cc = ClusterCentroids(ratio=ratio, random_state=RND_SEED)

    assert_equal(cc.ratio, ratio)


def test_cc_fit_single_class():
    """Test either if an error when there is a single class"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    cc = ClusterCentroids(ratio=ratio, random_state=RND_SEED)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_warns(UserWarning, cc.fit, X, y_single_class)


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
    assert_equal(cc.stats_c_[0], 3)
    assert_equal(cc.stats_c_[1], 7)


def test_sample_wrong_X():
    """Test either if an error is raised when X is different at fitting
    and sampling"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    cc = ClusterCentroids(ratio=ratio, random_state=RND_SEED)
    cc.fit(X, Y)
    assert_raises(RuntimeError, cc.sample,
                  np.random.random((100, 40)), np.array([0] * 50 + [1] * 50))


def test_sample_wt_fit():
    """Test either if an error is raised when sample is called before
    fitting"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    cc = ClusterCentroids(ratio=ratio, random_state=RND_SEED)
    assert_raises(RuntimeError, cc.sample, X, Y)


def test_fit_sample_auto():
    """Test fit and sample routines with auto ratio"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    cc = ClusterCentroids(ratio=ratio, random_state=RND_SEED)

    # Fit and sample
    X_resampled, y_resampled = cc.fit_sample(X, Y)

    X_gt = np.array([[0.92923648, 0.76103773], [0.47104475, 0.44386323],
                     [0.13347175, 0.12167502], [0.06738818, -0.529627],
                     [0.17901516, 0.69860992], [0.094035, -2.55298982]])
    y_gt = np.array([0, 0, 0, 1, 1, 1])
    assert_array_almost_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_fit_sample_half():
    """Test fit and sample routines with ratio of .5"""

    # Define the parameter for the under-sampling
    ratio = .5

    # Create the object
    cc = ClusterCentroids(ratio=ratio, random_state=RND_SEED)

    # Fit and sample
    X_resampled, y_resampled = cc.fit_sample(X, Y)

    X_gt = np.array([[0.92923648, 0.76103773], [0.47104475, 0.44386323],
                     [0.13347175, 0.12167502], [0.09125309, -0.85409574],
                     [0.19220316, 0.32337101], [0.094035, -2.55298982],
                     [0.20792588, 1.49407907], [0.04352327, -0.20515826],
                     [0.12372842, 0.6536186]])
    y_gt = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
    assert_array_almost_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_sample_wrong_X_dft_ratio():
    """Test either if an error is raised when X is different at fitting
    and sampling without ratio"""

    # Create the object
    cc = ClusterCentroids(random_state=RND_SEED)
    cc.fit(X, Y)
    assert_raises(RuntimeError, cc.sample,
                  np.random.random((100, 40)), np.array([0] * 50 + [1] * 50))


def test_continuous_error():
    """Test either if an error is raised when the target are continuous
    type"""

    # continuous case
    y = np.linspace(0, 1, 10)
    cc = ClusterCentroids(random_state=RND_SEED)
    assert_warns(UserWarning, cc.fit, X, y)


def test_multiclass_fit_sample():
    """Test fit sample method with multiclass target"""

    # Make y to be multiclass
    y = Y.copy()
    y[5] = 2
    y[6] = 2

    # Resample the data
    cc = ClusterCentroids(random_state=RND_SEED)
    X_resampled, y_resampled = cc.fit_sample(X, y)

    # Check the size of y
    count_y_res = Counter(y_resampled)
    assert_equal(count_y_res[0], 2)
    assert_equal(count_y_res[1], 2)
    assert_equal(count_y_res[2], 2)


def test_fit_sample_object():
    """Test fit and sample using a KMeans object"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    cluster = KMeans(random_state=RND_SEED)
    cc = ClusterCentroids(
        ratio=ratio, random_state=RND_SEED, estimator=cluster)

    # Fit and sample
    X_resampled, y_resampled = cc.fit_sample(X, Y)

    X_gt = np.array([[0.92923648, 0.76103773], [0.47104475, 0.44386323],
                     [0.13347175, 0.12167502], [0.06738818, -0.529627],
                     [0.17901516, 0.69860992], [0.094035, -2.55298982]])
    y_gt = np.array([0, 0, 0, 1, 1, 1])
    assert_array_almost_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_fit_sample_wrong_object():
    """Test fit and sample using a KMeans object"""

    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    cluster = 'rnd'
    cc = ClusterCentroids(
        ratio=ratio, random_state=RND_SEED, estimator=cluster)

    # Fit and sample
    assert_raises(ValueError, cc.fit_sample, X, Y)
