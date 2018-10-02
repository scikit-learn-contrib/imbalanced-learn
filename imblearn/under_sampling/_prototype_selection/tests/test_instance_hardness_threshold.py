"""Test the module ."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import pytest
import numpy as np

from sklearn.utils.testing import assert_array_equal
from sklearn.ensemble import GradientBoostingClassifier

from imblearn.under_sampling import InstanceHardnessThreshold

RND_SEED = 0
X = np.array([[-0.3879569, 0.6894251], [-0.09322739, 1.28177189], [
    -0.77740357, 0.74097941
], [0.91542919, -0.65453327], [-0.03852113, 0.40910479], [
    -0.43877303, 1.07366684
], [-0.85795321, 0.82980738], [-0.18430329, 0.52328473], [
    -0.30126957, -0.66268378
], [-0.65571327, 0.42412021], [-0.28305528, 0.30284991],
              [0.20246714, -0.34727125], [1.06446472, -1.09279772],
              [0.30543283, -0.02589502], [-0.00717161, 0.00318087]])
Y = np.array([0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0])
ESTIMATOR = GradientBoostingClassifier(random_state=RND_SEED)


def test_iht_init():
    sampling_strategy = 'auto'
    iht = InstanceHardnessThreshold(
        ESTIMATOR, sampling_strategy=sampling_strategy, random_state=RND_SEED)

    assert iht.sampling_strategy == sampling_strategy
    assert iht.random_state == RND_SEED


def test_iht_fit_resample():
    iht = InstanceHardnessThreshold(ESTIMATOR, random_state=RND_SEED)
    X_resampled, y_resampled = iht.fit_resample(X, Y)

    X_gt = np.array([[-0.3879569, 0.6894251], [0.91542919, -0.65453327], [
        -0.65571327, 0.42412021
    ], [1.06446472, -1.09279772], [0.30543283, -0.02589502], [
        -0.00717161, 0.00318087
    ], [-0.09322739, 1.28177189], [-0.77740357, 0.74097941],
                     [-0.43877303, 1.07366684], [-0.85795321, 0.82980738],
                     [-0.18430329, 0.52328473], [-0.28305528, 0.30284991]])
    y_gt = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


@pytest.mark.filterwarnings("ignore:'return_indices' is deprecated from 0.4")
def test_iht_fit_resample_with_indices():
    iht = InstanceHardnessThreshold(
        ESTIMATOR, return_indices=True, random_state=RND_SEED)
    X_resampled, y_resampled, idx_under = iht.fit_resample(X, Y)

    X_gt = np.array([[-0.3879569, 0.6894251], [0.91542919, -0.65453327], [
        -0.65571327, 0.42412021
    ], [1.06446472, -1.09279772], [0.30543283, -0.02589502], [
        -0.00717161, 0.00318087
    ], [-0.09322739, 1.28177189], [-0.77740357, 0.74097941],
                     [-0.43877303, 1.07366684], [-0.85795321, 0.82980738],
                     [-0.18430329, 0.52328473], [-0.28305528, 0.30284991]])
    y_gt = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    idx_gt = np.array([0, 3, 9, 12, 13, 14, 1, 2, 5, 6, 7, 10])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


def test_iht_fit_resample_half():
    sampling_strategy = {0: 6, 1: 8}
    iht = InstanceHardnessThreshold(
        ESTIMATOR, sampling_strategy=sampling_strategy, random_state=RND_SEED)
    X_resampled, y_resampled = iht.fit_resample(X, Y)

    X_gt = np.array([[-0.3879569, 0.6894251], [0.91542919, -0.65453327], [
        -0.65571327, 0.42412021
    ], [1.06446472, -1.09279772], [0.30543283, -0.02589502], [
        -0.00717161, 0.00318087
    ], [-0.09322739, 1.28177189], [-0.77740357, 0.74097941],
                     [-0.03852113, 0.40910479], [-0.43877303, 1.07366684],
                     [-0.85795321, 0.82980738], [-0.18430329, 0.52328473],
                     [-0.30126957, -0.66268378], [-0.28305528, 0.30284991]])
    y_gt = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_iht_fit_resample_class_obj():
    est = GradientBoostingClassifier(random_state=RND_SEED)
    iht = InstanceHardnessThreshold(estimator=est, random_state=RND_SEED)
    X_resampled, y_resampled = iht.fit_resample(X, Y)

    X_gt = np.array([[-0.3879569, 0.6894251], [0.91542919, -0.65453327], [
        -0.65571327, 0.42412021
    ], [1.06446472, -1.09279772], [0.30543283, -0.02589502], [
        -0.00717161, 0.00318087
    ], [-0.09322739, 1.28177189], [-0.77740357, 0.74097941],
                     [-0.43877303, 1.07366684], [-0.85795321, 0.82980738],
                     [-0.18430329, 0.52328473], [-0.28305528, 0.30284991]])
    y_gt = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_iht_fit_resample_wrong_class_obj():
    from sklearn.cluster import KMeans
    est = KMeans()
    iht = InstanceHardnessThreshold(estimator=est, random_state=RND_SEED)
    with pytest.raises(ValueError, match="Invalid parameter `estimator`"):
        iht.fit_resample(X, Y)
