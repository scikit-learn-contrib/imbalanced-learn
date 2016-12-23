"""Test the module ."""
from __future__ import print_function

import numpy as np
from numpy.testing import (assert_array_equal, assert_equal, assert_raises,
                           assert_warns)
from sklearn.utils.estimator_checks import check_estimator
from sklearn.ensemble import GradientBoostingClassifier

from imblearn.under_sampling import InstanceHardnessThreshold

# Generate a global dataset to use
RND_SEED = 0
X = np.array([[-0.3879569, 0.6894251], [-0.09322739, 1.28177189],
              [-0.77740357, 0.74097941], [0.91542919, -0.65453327],
              [-0.03852113, 0.40910479], [-0.43877303, 1.07366684],
              [-0.85795321, 0.82980738], [-0.18430329, 0.52328473],
              [-0.30126957, -0.66268378], [-0.65571327, 0.42412021],
              [-0.28305528, 0.30284991], [0.20246714, -0.34727125],
              [1.06446472, -1.09279772], [0.30543283, -0.02589502],
              [-0.00717161, 0.00318087]])
Y = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0])
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
    ratio = 0.7
    est = 'rnd'
    iht = InstanceHardnessThreshold(
        estimator=est, ratio=ratio, random_state=RND_SEED)
    assert_raises(NotImplementedError, iht.fit_sample, X, Y)


def test_iht_init():
    """Test the initialisation of the object"""

    # Define a ratio
    ratio = 'auto'
    iht = InstanceHardnessThreshold(
        ESTIMATOR, ratio=ratio, random_state=RND_SEED)

    assert_equal(iht.ratio, ratio)
    assert_equal(iht.random_state, RND_SEED)


def test_iht_fit_single_class():
    """Test either if an error when there is a single class"""

    # Create the object
    iht = InstanceHardnessThreshold(ESTIMATOR, random_state=RND_SEED)
    # Resample the data
    # Create a wrong y
    y_single_class = np.zeros((X.shape[0], ))
    assert_warns(UserWarning, iht.fit, X, y_single_class)


def test_iht_fit_invalid_ratio():
    """Test either if an error is raised when the balancing ratio to fit is
    smaller than the one of the data"""

    # Create the object
    ratio = 1. / 10000.
    iht = InstanceHardnessThreshold(
        ESTIMATOR, ratio=ratio, random_state=RND_SEED)
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
    assert_equal(iht.stats_c_[0], 6)
    assert_equal(iht.stats_c_[1], 9)


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

    X_gt = np.array([[-0.3879569, 0.6894251], [-0.09322739, 1.28177189],
                     [-0.77740357, 0.74097941], [0.91542919, -0.65453327],
                     [-0.43877303, 1.07366684], [-0.85795321, 0.82980738],
                     [-0.18430329, 0.52328473], [-0.65571327, 0.42412021],
                     [-0.28305528, 0.30284991], [1.06446472, -1.09279772],
                     [0.30543283, -0.02589502], [-0.00717161, 0.00318087]])
    y_gt = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_iht_fit_sample_with_indices():
    """Test the fit sample routine with indices support"""

    # Resample the data
    iht = InstanceHardnessThreshold(
        ESTIMATOR, return_indices=True, random_state=RND_SEED)
    X_resampled, y_resampled, idx_under = iht.fit_sample(X, Y)

    X_gt = np.array([[-0.3879569, 0.6894251], [-0.09322739, 1.28177189],
                     [-0.77740357, 0.74097941], [0.91542919, -0.65453327],
                     [-0.43877303, 1.07366684], [-0.85795321, 0.82980738],
                     [-0.18430329, 0.52328473], [-0.65571327, 0.42412021],
                     [-0.28305528, 0.30284991], [1.06446472, -1.09279772],
                     [0.30543283, -0.02589502], [-0.00717161, 0.00318087]])
    y_gt = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0])
    idx_gt = np.array([0, 1, 2, 3, 5, 6, 7, 9, 10, 12, 13, 14])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


def test_iht_fit_sample_half():
    """Test the fit sample routine with a 0.5 ratio"""

    # Resample the data
    ratio = 0.7
    iht = InstanceHardnessThreshold(
        ESTIMATOR, ratio=ratio, random_state=RND_SEED)
    X_resampled, y_resampled = iht.fit_sample(X, Y)

    X_gt = np.array([[-0.3879569, 0.6894251], [-0.09322739, 1.28177189],
                     [-0.77740357, 0.74097941], [0.91542919, -0.65453327],
                     [-0.03852113, 0.40910479], [-0.43877303, 1.07366684],
                     [-0.85795321, 0.82980738], [-0.18430329, 0.52328473],
                     [-0.30126957, -0.66268378], [-0.65571327, 0.42412021],
                     [-0.28305528, 0.30284991], [1.06446472, -1.09279772],
                     [0.30543283, -0.02589502], [-0.00717161, 0.00318087]])
    y_gt = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_iht_fit_sample_knn():
    """Test the fit sample routine with knn"""

    # Resample the data
    est = 'knn'
    iht = InstanceHardnessThreshold(est, random_state=RND_SEED)
    X_resampled, y_resampled = iht.fit_sample(X, Y)

    X_gt = np.array([[-0.3879569, 0.6894251], [-0.09322739, 1.28177189],
                     [-0.77740357, 0.74097941], [0.91542919, -0.65453327],
                     [-0.43877303, 1.07366684], [-0.85795321, 0.82980738],
                     [-0.30126957, -0.66268378], [-0.65571327, 0.42412021],
                     [0.20246714, -0.34727125], [1.06446472, -1.09279772],
                     [0.30543283, -0.02589502], [-0.00717161, 0.00318087]])
    y_gt = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_iht_fit_sample_decision_tree():
    """Test the fit sample routine with decision-tree"""

    # Resample the data
    est = 'decision-tree'
    iht = InstanceHardnessThreshold(est, random_state=RND_SEED)
    X_resampled, y_resampled = iht.fit_sample(X, Y)

    X_gt = np.array([[-0.3879569, 0.6894251], [-0.09322739, 1.28177189],
                     [-0.77740357, 0.74097941], [0.91542919, -0.65453327],
                     [-0.43877303, 1.07366684], [-0.85795321, 0.82980738],
                     [-0.18430329, 0.52328473], [-0.65571327, 0.42412021],
                     [-0.28305528, 0.30284991], [1.06446472, -1.09279772],
                     [0.30543283, -0.02589502], [-0.00717161, 0.00318087]])
    y_gt = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_iht_fit_sample_random_forest():
    """Test the fit sample routine with random forest"""

    # Resample the data
    est = 'random-forest'
    iht = InstanceHardnessThreshold(est, random_state=RND_SEED)
    X_resampled, y_resampled = iht.fit_sample(X, Y)

    X_gt = np.array([[-0.3879569, 0.6894251], [-0.09322739, 1.28177189],
                     [-0.77740357, 0.74097941], [0.91542919, -0.65453327],
                     [-0.03852113, 0.40910479], [-0.43877303, 1.07366684],
                     [-0.85795321, 0.82980738], [-0.18430329, 0.52328473],
                     [-0.65571327, 0.42412021], [-0.28305528, 0.30284991],
                     [1.06446472, -1.09279772], [0.30543283, -0.02589502],
                     [-0.00717161, 0.00318087]])
    y_gt = np.array([0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_iht_fit_sample_adaboost():
    """Test the fit sample routine with adaboost"""

    # Resample the data
    est = 'adaboost'
    iht = InstanceHardnessThreshold(est, random_state=RND_SEED)
    X_resampled, y_resampled = iht.fit_sample(X, Y)

    X_gt = np.array([[-0.3879569, 0.6894251], [-0.09322739, 1.28177189],
                     [-0.77740357, 0.74097941], [0.91542919, -0.65453327],
                     [-0.43877303, 1.07366684], [-0.85795321, 0.82980738],
                     [-0.18430329, 0.52328473], [-0.65571327, 0.42412021],
                     [-0.28305528, 0.30284991], [1.06446472, -1.09279772],
                     [0.30543283, -0.02589502], [-0.00717161, 0.00318087]])
    y_gt = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_iht_fit_sample_gradient_boosting():
    """Test the fit sample routine with gradient boosting"""

    # Resample the data
    est = 'gradient-boosting'
    iht = InstanceHardnessThreshold(est, random_state=RND_SEED)
    X_resampled, y_resampled = iht.fit_sample(X, Y)

    X_gt = np.array([[-0.3879569, 0.6894251], [-0.09322739, 1.28177189],
                     [-0.77740357, 0.74097941], [0.91542919, -0.65453327],
                     [-0.43877303, 1.07366684], [-0.85795321, 0.82980738],
                     [-0.18430329, 0.52328473], [-0.65571327, 0.42412021],
                     [-0.28305528, 0.30284991], [1.06446472, -1.09279772],
                     [0.30543283, -0.02589502], [-0.00717161, 0.00318087]])
    y_gt = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_iht_fit_sample_linear_svm():
    """Test the fit sample routine with linear SVM"""

    # Resample the data
    est = 'linear-svm'
    iht = InstanceHardnessThreshold(est, random_state=RND_SEED)
    X_resampled, y_resampled = iht.fit_sample(X, Y)

    X_gt = np.array([[-0.3879569, 0.6894251], [-0.09322739, 1.28177189],
                     [-0.77740357, 0.74097941], [0.91542919, -0.65453327],
                     [-0.03852113, 0.40910479], [-0.43877303, 1.07366684],
                     [-0.18430329, 0.52328473], [-0.65571327, 0.42412021],
                     [-0.28305528, 0.30284991], [1.06446472, -1.09279772],
                     [0.30543283, -0.02589502], [-0.00717161, 0.00318087]])
    y_gt = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_iht_sample_wrong_X():
    """Test either if an error is raised when X is different at fitting
    and sampling"""

    # Create the object
    iht = InstanceHardnessThreshold(random_state=RND_SEED)
    iht.fit(X, Y)
    assert_raises(RuntimeError, iht.sample,
                  np.random.random((100, 40)), np.array([0] * 50 + [1] * 50))


def test_multiclass_error():
    """ Test either if an error is raised when the target are not binary
    type. """

    # continuous case
    y = np.linspace(0, 1, 15)
    iht = InstanceHardnessThreshold(random_state=RND_SEED)
    assert_warns(UserWarning, iht.fit, X, y)

    # multiclass case
    y = np.array([0] * 10 + [1] * 3 + [2] * 2)
    iht = InstanceHardnessThreshold(random_state=RND_SEED)
    assert_warns(UserWarning, iht.fit, X, y)


def test_iht_fit_sample_class_obj():
    """Test the fit sample routine passing a classifiermixin object"""

    # Resample the data
    est = GradientBoostingClassifier(random_state=RND_SEED)
    iht = InstanceHardnessThreshold(estimator=est, random_state=RND_SEED)
    X_resampled, y_resampled = iht.fit_sample(X, Y)

    X_gt = np.array([[-0.3879569, 0.6894251], [-0.09322739, 1.28177189],
                     [-0.77740357, 0.74097941], [0.91542919, -0.65453327],
                     [-0.43877303, 1.07366684], [-0.85795321, 0.82980738],
                     [-0.18430329, 0.52328473], [-0.65571327, 0.42412021],
                     [-0.28305528, 0.30284991], [1.06446472, -1.09279772],
                     [0.30543283, -0.02589502], [-0.00717161, 0.00318087]])
    y_gt = np.array([0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_iht_fit_sample_wrong_class_obj():
    """Test either if an error is raised while passing a wrong classifier
    object"""

    # Resample the data
    from sklearn.cluster import KMeans
    est = KMeans()
    iht = InstanceHardnessThreshold(estimator=est, random_state=RND_SEED)
    assert_raises(ValueError, iht.fit_sample, X, Y)
