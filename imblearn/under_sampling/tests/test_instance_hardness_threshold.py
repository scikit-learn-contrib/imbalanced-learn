"""Test the module ."""
from __future__ import print_function

import os

import numpy as np
from numpy.testing import assert_raises
from numpy.testing import assert_equal
from numpy.testing import assert_array_equal
from numpy.testing import assert_warns

from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import check_estimator

from imblearn.under_sampling import InstanceHardnessThreshold


# Generate a global dataset to use
RND_SEED = 0
X, Y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9],
                           n_informative=3, n_redundant=1, flip_y=0,
                           n_features=20, n_clusters_per_class=1,
                           n_samples=5000, random_state=RND_SEED)
ESTIMATOR = 'gradient-boosting'


def test_iht_sk_estimator():
    """Test the sklearn estimator compatibility"""
    check_estimator(InstanceHardnessThreshold)


def test_iht_bad_ratio():
    """Test either if an error is raised with a wrong decimal value for
    the ratio"""

    # Define a negative ratio
    ratio = -1.0
    iht = InstanceHardnessThreshold(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, iht.fit, X, Y)

    # Define a ratio greater than 1
    ratio = 100.0
    iht = InstanceHardnessThreshold(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, iht.fit, X, Y)

    # Define ratio as an unknown string
    ratio = 'rnd'
    iht = InstanceHardnessThreshold(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, iht.fit, X, Y)

    # Define ratio as a list which is not supported
    ratio = [.5, .5]
    iht = InstanceHardnessThreshold(ratio=ratio, random_state=RND_SEED)
    assert_raises(ValueError, iht.fit, X, Y)


def test_iht_wrong_estimator():
    """Test either if an error is raised when the estimator is unknown"""

    # Resample the data
    ratio = 0.5
    est = 'rnd'
    iht = InstanceHardnessThreshold(estimator=est, ratio=ratio,
                                    random_state=RND_SEED)
    assert_raises(NotImplementedError, iht.fit_sample, X, Y)


def test_iht_init():
    """Test the initialisation of the object"""

    # Define a ratio
    ratio = 'auto'
    iht = InstanceHardnessThreshold(ESTIMATOR, ratio=ratio,
                                    random_state=RND_SEED)

    assert_equal(iht.ratio, ratio)
    assert_equal(iht.random_state, RND_SEED)


def test_iht_fit_single_class():
    """Test either if an error when there is a single class"""

    # Create the object
    iht = InstanceHardnessThreshold(ESTIMATOR, random_state=RND_SEED)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_warns(RuntimeWarning, iht.fit, X, y_single_class)


def test_iht_fit_invalid_ratio():
    """Test either if an error is raised when the balancing ratio to fit is
    smaller than the one of the data"""

    # Create the object
    ratio = 1. / 10000.
    iht = InstanceHardnessThreshold(ESTIMATOR, ratio=ratio,
                                    random_state=RND_SEED)
    # Fit the data
    assert_raises(RuntimeError, iht.fit, X, Y)


def test_iht_fit():
    """Test the fitting method"""

    # Create the object
    iht = InstanceHardnessThreshold(ESTIMATOR, random_state=RND_SEED)
    # Fit the data
    iht.fit(X, Y)

    # Check if the data information have been computed
    assert_equal(iht.min_c_, 0)
    assert_equal(iht.maj_c_, 1)
    assert_equal(iht.stats_c_[0], 500)
    assert_equal(iht.stats_c_[1], 4500)


def test_iht_sample_wt_fit():
    """Test either if an error is raised when sample is called before
    fitting"""

    # Create the object
    iht = InstanceHardnessThreshold(ESTIMATOR, random_state=RND_SEED)
    assert_raises(RuntimeError, iht.sample, X, Y)


def test_iht_fit_sample():
    """Test the fit sample routine"""

    # Resample the data
    iht = InstanceHardnessThreshold(ESTIMATOR, random_state=RND_SEED)
    X_resampled, y_resampled = iht.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'iht_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'iht_y.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_iht_fit_sample_with_indices():
    """Test the fit sample routine with indices support"""

    # Resample the data
    iht = InstanceHardnessThreshold(ESTIMATOR, return_indices=True,
                                    random_state=RND_SEED)
    X_resampled, y_resampled, idx_under = iht.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'iht_x.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'iht_y.npy'))
    idx_gt = np.load(os.path.join(currdir, 'data', 'iht_idx.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


def test_iht_fit_sample_half():
    """Test the fit sample routine with a 0.5 ratio"""

    # Resample the data
    ratio = 0.5
    iht = InstanceHardnessThreshold(ESTIMATOR, ratio=ratio,
                                    random_state=RND_SEED)
    X_resampled, y_resampled = iht.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'iht_x_05.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'iht_y_05.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_iht_fit_sample_knn():
    """Test the fit sample routine with knn"""

    # Resample the data
    est = 'knn'
    iht = InstanceHardnessThreshold(est, random_state=RND_SEED)
    X_resampled, y_resampled = iht.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'iht_x_knn.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'iht_y_knn.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_iht_fit_sample_decision_tree():
    """Test the fit sample routine with decision-tree"""

    # Resample the data
    est = 'decision-tree'
    iht = InstanceHardnessThreshold(est, random_state=RND_SEED)
    X_resampled, y_resampled = iht.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'iht_x_dt.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'iht_y_dt.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_iht_fit_sample_random_forest():
    """Test the fit sample routine with random forest"""

    # Resample the data
    est = 'random-forest'
    iht = InstanceHardnessThreshold(est, random_state=RND_SEED)
    X_resampled, y_resampled = iht.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'iht_x_rf.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'iht_y_rf.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_iht_fit_sample_adaboost():
    """Test the fit sample routine with adaboost"""

    # Resample the data
    est = 'adaboost'
    iht = InstanceHardnessThreshold(est, random_state=RND_SEED)
    X_resampled, y_resampled = iht.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'iht_x_adb.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'iht_y_adb.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_iht_fit_sample_gradient_boosting():
    """Test the fit sample routine with gradient boosting"""

    # Resample the data
    est = 'gradient-boosting'
    iht = InstanceHardnessThreshold(est, random_state=RND_SEED)
    X_resampled, y_resampled = iht.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'iht_x_gb.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'iht_y_gb.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_iht_fit_sample_linear_svm():
    """Test the fit sample routine with linear SVM"""

    # Resample the data
    est = 'linear-svm'
    iht = InstanceHardnessThreshold(est, random_state=RND_SEED)
    X_resampled, y_resampled = iht.fit_sample(X, Y)

    currdir = os.path.dirname(os.path.abspath(__file__))
    X_gt = np.load(os.path.join(currdir, 'data', 'iht_x_svm.npy'))
    y_gt = np.load(os.path.join(currdir, 'data', 'iht_y_svm.npy'))
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_iht_sample_wrong_X():
    """Test either if an error is raised when X is different at fitting
    and sampling"""

    # Create the object
    iht = InstanceHardnessThreshold(random_state=RND_SEED)
    iht.fit(X, Y)
    assert_raises(RuntimeError, iht.sample, np.random.random((100, 40)),
                  np.array([0] * 50 + [1] * 50))
