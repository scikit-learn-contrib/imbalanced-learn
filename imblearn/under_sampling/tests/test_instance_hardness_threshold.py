"""Test the module ."""
from __future__ import print_function

import numpy as np
from numpy.testing import (assert_array_equal, assert_equal, assert_raises,
                           assert_raises_regex)
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


def test_iht_wrong_estimator():
    # Resample the data
    ratio = 0.7
    est = 'rnd'
    iht = InstanceHardnessThreshold(
        estimator=est, ratio=ratio, random_state=RND_SEED)
    assert_raises(NotImplementedError, iht.fit_sample, X, Y)


def test_iht_init():
    # Define a ratio
    ratio = 'auto'
    iht = InstanceHardnessThreshold(
        ESTIMATOR, ratio=ratio, random_state=RND_SEED)

    assert_equal(iht.ratio, ratio)
    assert_equal(iht.random_state, RND_SEED)


def test_iht_fit_sample():
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


def test_iht_fit_sample_class_obj():
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
    # Resample the data
    from sklearn.cluster import KMeans
    est = KMeans()
    iht = InstanceHardnessThreshold(estimator=est, random_state=RND_SEED)
    assert_raises_regex(ValueError, "Invalid parameter `estimator`",
                        iht.fit_sample, X, Y)
