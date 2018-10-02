"""Test the module condensed nearest neighbour."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import pytest
import numpy as np

from sklearn.utils.testing import assert_array_equal
from sklearn.neighbors import KNeighborsClassifier

from imblearn.under_sampling import CondensedNearestNeighbour

RND_SEED = 0
X = np.array([[2.59928271, 0.93323465], [0.25738379, 0.95564169], [
    1.42772181, 0.526027
], [1.92365863, 0.82718767], [-0.10903849,
                              -0.12085181], [-0.284881, -0.62730973],
              [0.57062627, 1.19528323], [0.03394306,
                                         0.03986753], [0.78318102, 2.59153329],
              [0.35831463, 1.33483198], [-0.14313184, -1.0412815], [
                  0.01936241, 0.17799828
              ], [-1.25020462, -0.40402054], [-0.09816301, -0.74662486], [
                  -0.01252787, 0.34102657
              ], [0.52726792, -0.38735648], [0.2821046, -0.07862747], [
                  0.05230552, 0.09043907
              ], [0.15198585, 0.12512646], [0.70524765, 0.39816382]])
Y = np.array([1, 2, 1, 1, 0, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 1, 2, 1])


def test_cnn_init():
    cnn = CondensedNearestNeighbour(random_state=RND_SEED)

    assert cnn.n_seeds_S == 1
    assert cnn.n_jobs == 1


def test_cnn_fit_resample():
    cnn = CondensedNearestNeighbour(random_state=RND_SEED)
    X_resampled, y_resampled = cnn.fit_resample(X, Y)

    X_gt = np.array([[-0.10903849, -0.12085181], [0.01936241, 0.17799828], [
        0.05230552, 0.09043907
    ], [-1.25020462, -0.40402054], [0.70524765,
                                    0.39816382], [0.35831463, 1.33483198],
                     [-0.284881, -0.62730973], [0.03394306, 0.03986753],
                     [-0.01252787, 0.34102657], [0.15198585, 0.12512646]])
    y_gt = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


@pytest.mark.filterwarnings("ignore:'return_indices' is deprecated from 0.4")
def test_cnn_fit_resample_with_indices():
    cnn = CondensedNearestNeighbour(return_indices=True, random_state=RND_SEED)
    X_resampled, y_resampled, idx_under = cnn.fit_resample(X, Y)

    X_gt = np.array([[-0.10903849, -0.12085181], [0.01936241, 0.17799828], [
        0.05230552, 0.09043907
    ], [-1.25020462, -0.40402054], [0.70524765,
                                    0.39816382], [0.35831463, 1.33483198],
                     [-0.284881, -0.62730973], [0.03394306, 0.03986753],
                     [-0.01252787, 0.34102657], [0.15198585, 0.12512646]])
    y_gt = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    idx_gt = np.array([4, 11, 17, 12, 19, 9, 5, 7, 14, 18])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


def test_cnn_fit_resample_with_object():
    knn = KNeighborsClassifier(n_neighbors=1)
    cnn = CondensedNearestNeighbour(random_state=RND_SEED, n_neighbors=knn)
    X_resampled, y_resampled = cnn.fit_resample(X, Y)

    X_gt = np.array([[-0.10903849, -0.12085181], [0.01936241, 0.17799828], [
        0.05230552, 0.09043907
    ], [-1.25020462, -0.40402054], [0.70524765,
                                    0.39816382], [0.35831463, 1.33483198],
                     [-0.284881, -0.62730973], [0.03394306, 0.03986753],
                     [-0.01252787, 0.34102657], [0.15198585, 0.12512646]])
    y_gt = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)

    cnn = CondensedNearestNeighbour(random_state=RND_SEED, n_neighbors=1)
    X_resampled, y_resampled = cnn.fit_resample(X, Y)
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_cnn_fit_resample_with_wrong_object():
    knn = 'rnd'
    cnn = CondensedNearestNeighbour(random_state=RND_SEED, n_neighbors=knn)
    with pytest.raises(ValueError, match="has to be a int or an "):
        cnn.fit_resample(X, Y)
