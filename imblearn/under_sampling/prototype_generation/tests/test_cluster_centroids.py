"""Test the module cluster centroids."""
from __future__ import print_function

from collections import Counter

import numpy as np
from scipy import sparse
from pytest import raises

from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import assert_array_equal
from sklearn.cluster import KMeans

from imblearn.under_sampling import ClusterCentroids

RND_SEED = 0
X = np.array([[0.04352327, -0.20515826], [0.92923648, 0.76103773],
              [0.20792588, 1.49407907], [0.47104475, 0.44386323],
              [0.22950086, 0.33367433], [0.15490546, 0.3130677],
              [0.09125309, -0.85409574], [0.12372842, 0.6536186],
              [0.13347175, 0.12167502], [0.094035, -2.55298982]])
Y = np.array([1, 0, 1, 0, 1, 1, 1, 1, 0, 1])
R_TOL = 1e-4


def test_fit_sample_check_voting():
    cc = ClusterCentroids(random_state=RND_SEED)
    cc.fit_sample(X, Y)
    assert cc.voting_ == 'soft'
    cc = ClusterCentroids(random_state=RND_SEED)
    cc.fit_sample(sparse.csr_matrix(X), Y)
    assert cc.voting_ == 'hard'


def test_fit_sample_auto():
    ratio = 'auto'
    cc = ClusterCentroids(ratio=ratio, random_state=RND_SEED)
    X_resampled, y_resampled = cc.fit_sample(X, Y)
    X_gt = np.array([[0.92923648, 0.76103773],
                     [0.47104475, 0.44386323],
                     [0.13347175, 0.12167502],
                     [0.06738818, -0.529627],
                     [0.17901516, 0.69860992],
                     [0.094035, -2.55298982]])
    y_gt = np.array([0, 0, 0, 1, 1, 1])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_fit_sample_half():
    ratio = .5
    cc = ClusterCentroids(ratio=ratio, random_state=RND_SEED)
    X_resampled, y_resampled = cc.fit_sample(X, Y)
    X_gt = np.array([[0.92923648, 0.76103773],
                     [0.47104475, 0.44386323],
                     [0.13347175, 0.12167502],
                     [0.09125309, -0.85409574],
                     [0.19220316, 0.32337101],
                     [0.094035, -2.55298982],
                     [0.20792588, 1.49407907],
                     [0.04352327, -0.20515826],
                     [0.12372842, 0.6536186]])
    y_gt = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_multiclass_fit_sample():
    y = Y.copy()
    y[5] = 2
    y[6] = 2
    cc = ClusterCentroids(random_state=RND_SEED)
    X_resampled, y_resampled = cc.fit_sample(X, y)
    count_y_res = Counter(y_resampled)
    assert count_y_res[0] == 2
    assert count_y_res[1] == 2
    assert count_y_res[2] == 2


def test_fit_sample_object():
    ratio = 'auto'
    cluster = KMeans(random_state=RND_SEED)
    cc = ClusterCentroids(
        ratio=ratio, random_state=RND_SEED, estimator=cluster)

    X_resampled, y_resampled = cc.fit_sample(X, Y)
    X_gt = np.array([[0.92923648, 0.76103773],
                     [0.47104475, 0.44386323],
                     [0.13347175, 0.12167502],
                     [0.06738818, -0.529627],
                     [0.17901516, 0.69860992],
                     [0.094035, -2.55298982]])
    y_gt = np.array([0, 0, 0, 1, 1, 1])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_fit_hard_voting():
    ratio = 'auto'
    voting = 'hard'
    cluster = KMeans(random_state=RND_SEED)
    cc = ClusterCentroids(
        ratio=ratio, random_state=RND_SEED, estimator=cluster,
        voting=voting)

    X_resampled, y_resampled = cc.fit_sample(X, Y)
    X_gt = np.array([[0.92923648, 0.76103773],
                     [0.47104475, 0.44386323],
                     [0.13347175, 0.12167502],
                     [0.09125309, -0.85409574],
                     [0.12372842, 0.6536186],
                     [0.094035, -2.55298982]])
    y_gt = np.array([0, 0, 0, 1, 1, 1])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)
    for x in X_resampled:
        assert np.any(np.all(x == X, axis=1))


def test_fit_sample_error():
    ratio = 'auto'
    cluster = 'rnd'
    cc = ClusterCentroids(
        ratio=ratio, random_state=RND_SEED, estimator=cluster)
    with raises(ValueError, match="has to be a KMeans clustering"):
        cc.fit_sample(X, Y)

    voting = 'unknown'
    cc = ClusterCentroids(ratio=ratio, voting=voting, random_state=RND_SEED)
    with raises(ValueError, match="needs to be one of"):
        cc.fit_sample(X, Y)
