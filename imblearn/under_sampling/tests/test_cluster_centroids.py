"""Test the module cluster centroids."""
from __future__ import print_function

from collections import Counter

import numpy as np
from numpy.testing import (assert_allclose, assert_array_equal,
                           assert_equal, assert_raises_regex)
from sklearn.cluster import KMeans

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
R_TOL = 1e-4


def test_init():
    # Define a ratio
    ratio = 1.
    cc = ClusterCentroids(ratio=ratio, random_state=RND_SEED)

    assert_equal(cc.ratio, ratio)


def test_fit_sample_auto():
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
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_fit_sample_half():
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
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_multiclass_fit_sample():
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
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_fit_sample_wrong_object():
    # Define the parameter for the under-sampling
    ratio = 'auto'

    # Create the object
    cluster = 'rnd'
    cc = ClusterCentroids(
        ratio=ratio, random_state=RND_SEED, estimator=cluster)

    # Fit and sample
    assert_raises_regex(ValueError, "has to be a KMeans clustering",
                        cc.fit_sample, X, Y)
