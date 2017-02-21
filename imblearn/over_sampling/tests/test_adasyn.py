"""Test the module under sampler."""
from __future__ import print_function

import numpy as np
from numpy.testing import (assert_allclose, assert_array_equal,
                           assert_equal, assert_raises, assert_warns)
from sklearn.utils.estimator_checks import check_estimator
from sklearn.neighbors import NearestNeighbors

from imblearn.over_sampling import ADASYN

# Generate a global dataset to use
RND_SEED = 0
X = np.array([[0.11622591, -0.0317206], [0.77481731, 0.60935141],
              [1.25192108, -0.22367336], [0.53366841, -0.30312976],
              [1.52091956, -0.49283504], [-0.28162401, -2.10400981],
              [0.83680821, 1.72827342], [0.3084254, 0.33299982],
              [0.70472253, -0.73309052], [0.28893132, -0.38761769],
              [1.15514042, 0.0129463], [0.88407872, 0.35454207],
              [1.31301027, -0.92648734], [-1.11515198, -0.93689695],
              [-0.18410027, -0.45194484], [0.9281014, 0.53085498],
              [-0.14374509, 0.27370049], [-0.41635887, -0.38299653],
              [0.08711622, 0.93259929], [1.70580611, -0.11219234]])
Y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
R_TOL = 1e-4

def test_ada_sk_estimator():
    """Test the sklearn estimator compatibility"""
    check_estimator(ADASYN)


def test_ada_bad_ratio():
    """Test either if an error is raised with a wrong decimal value for
    the ratio"""

    # Define a negative ratio
    ratio = -1.0
    ada = ADASYN(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, ada.fit, X, Y)

    # Define a ratio greater than 1
    ratio = 100.0
    ada = ADASYN(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, ada.fit, X, Y)

    # Define ratio as an unknown string
    ratio = 'rnd'
    ada = ADASYN(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, ada.fit, X, Y)

    # Define ratio as a list which is not supported
    ratio = [.5, .5]
    ada = ADASYN(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, ada.fit, X, Y)


def test_ada_init():
    """Test the initialisation of the object"""

    # Define a ratio
    ratio = 'auto'
    ada = ADASYN(ratio=ratio, random_state=RND_SEED)

    assert_equal(ada.random_state, RND_SEED)


def test_ada_fit_single_class():
    """Test either if an error when there is a single class"""

    # Create the object
    ada = ADASYN(random_state=RND_SEED)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_warns(UserWarning, ada.fit, X, y_single_class)


def test_ada_fit_invalid_ratio():
    """Test either if an error is raised when the balancing ratio to fit is
    smaller than the one of the data"""

    # Create the object
    ratio = 1. / 10000.
    ada = ADASYN(ratio=ratio, random_state=RND_SEED)
    # Fit the data
    assert_raises(RuntimeError, ada.fit, X, Y)


def test_ada_fit():
    """Test the fitting method"""

    # Create the object
    ada = ADASYN(random_state=RND_SEED)
    # Fit the data
    ada.fit(X, Y)

    # Check if the data information have been computed
    assert_equal(ada.min_c_, 0)
    assert_equal(ada.maj_c_, 1)
    assert_equal(ada.stats_c_[0], 8)
    assert_equal(ada.stats_c_[1], 12)


def test_ada_sample_wt_fit():
    """Test either if an error is raised when sample is called before
    fitting"""

    # Create the object
    ada = ADASYN(random_state=RND_SEED)
    assert_raises(RuntimeError, ada.sample, X, Y)


def test_ada_fit_sample():
    """Test the fit sample routine"""

    # Resample the data
    ada = ADASYN(random_state=RND_SEED)
    X_resampled, y_resampled = ada.fit_sample(X, Y)

    X_gt = np.array([[0.11622591, -0.0317206], [0.77481731, 0.60935141],
                     [1.25192108, -0.22367336], [0.53366841, -0.30312976],
                     [1.52091956, -0.49283504], [-0.28162401, -2.10400981],
                     [0.83680821, 1.72827342], [0.3084254, 0.33299982],
                     [0.70472253, -0.73309052], [0.28893132, -0.38761769],
                     [1.15514042, 0.0129463], [0.88407872, 0.35454207],
                     [1.31301027, -0.92648734], [-1.11515198, -0.93689695],
                     [-0.18410027, -0.45194484], [0.9281014, 0.53085498],
                     [-0.14374509, 0.27370049], [-0.41635887, -0.38299653],
                     [0.08711622, 0.93259929], [1.70580611, -0.11219234],
                     [-0.06182085, -0.28084828], [0.38614986, -0.35405599],
                     [0.39635544, 0.33629036], [-0.24027923, 0.04116021]])
    y_gt = np.array([
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0
    ])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_ada_fit_sample_half():
    """Test the fit sample routine with a 0.5 ratio"""

    # Resample the data
    ratio = 0.8
    ada = ADASYN(ratio=ratio, random_state=RND_SEED)
    X_resampled, y_resampled = ada.fit_sample(X, Y)

    X_gt = np.array([[0.11622591, -0.0317206], [0.77481731, 0.60935141],
                     [1.25192108, -0.22367336], [0.53366841, -0.30312976],
                     [1.52091956, -0.49283504], [-0.28162401, -2.10400981],
                     [0.83680821, 1.72827342], [0.3084254, 0.33299982],
                     [0.70472253, -0.73309052], [0.28893132, -0.38761769],
                     [1.15514042, 0.0129463], [0.88407872, 0.35454207],
                     [1.31301027, -0.92648734], [-1.11515198, -0.93689695],
                     [-0.18410027, -0.45194484], [0.9281014, 0.53085498],
                     [-0.14374509, 0.27370049], [-0.41635887, -0.38299653],
                     [0.08711622, 0.93259929], [1.70580611, -0.11219234]])
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_wrong_X():
    """Test either if an error is raised when X is different at fitting
    and sampling"""

    # Create the object
    ada = ADASYN(random_state=RND_SEED)
    ada.fit(X, Y)
    assert_raises(RuntimeError, ada.sample,
                  np.random.random((100, 40)), np.array([0] * 50 + [1] * 50))


def test_multiclass_error():
    """ Test either if an error is raised when the target are not binary
    type. """

    # continuous case
    y = np.linspace(0, 1, 20)
    ada = ADASYN(random_state=RND_SEED)
    assert_warns(UserWarning, ada.fit, X, y)

    # multiclass case
    y = np.array([0] * 3 + [1] * 2 + [2] * 15)
    ada = ADASYN(random_state=RND_SEED)
    assert_warns(UserWarning, ada.fit, X, y)


def test_ada_fit_sample_nn_obj():
    """Test fit-sample with nn object"""

    # Resample the data
    nn = NearestNeighbors(n_neighbors=6)
    ada = ADASYN(random_state=RND_SEED, n_neighbors=nn)
    X_resampled, y_resampled = ada.fit_sample(X, Y)

    X_gt = np.array([[0.11622591, -0.0317206], [0.77481731, 0.60935141],
                     [1.25192108, -0.22367336], [0.53366841, -0.30312976],
                     [1.52091956, -0.49283504], [-0.28162401, -2.10400981],
                     [0.83680821, 1.72827342], [0.3084254, 0.33299982],
                     [0.70472253, -0.73309052], [0.28893132, -0.38761769],
                     [1.15514042, 0.0129463], [0.88407872, 0.35454207],
                     [1.31301027, -0.92648734], [-1.11515198, -0.93689695],
                     [-0.18410027, -0.45194484], [0.9281014, 0.53085498],
                     [-0.14374509, 0.27370049], [-0.41635887, -0.38299653],
                     [0.08711622, 0.93259929], [1.70580611, -0.11219234],
                     [-0.06182085, -0.28084828], [0.38614986, -0.35405599],
                     [0.39635544, 0.33629036], [-0.24027923, 0.04116021]])
    y_gt = np.array([
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0
    ])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_ada_wrong_nn_obj():
    """Test either if an error is raised while passing a wrong NN object"""

    # Resample the data
    nn = 'rnd'
    ada = ADASYN(random_state=RND_SEED, n_neighbors=nn)
    assert_raises(ValueError, ada.fit_sample, X, Y)
