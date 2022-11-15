"""Test the module SMOTE."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numpy as np
import pytest

from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_array_equal
from sklearn.neighbors import NearestNeighbors

from imblearn.over_sampling import SMOTE


RND_SEED = 0
X = np.array(
    [
        [0.11622591, -0.0317206],
        [0.77481731, 0.60935141],
        [1.25192108, -0.22367336],
        [0.53366841, -0.30312976],
        [1.52091956, -0.49283504],
        [-0.28162401, -2.10400981],
        [0.83680821, 1.72827342],
        [0.3084254, 0.33299982],
        [0.70472253, -0.73309052],
        [0.28893132, -0.38761769],
        [1.15514042, 0.0129463],
        [0.88407872, 0.35454207],
        [1.31301027, -0.92648734],
        [-1.11515198, -0.93689695],
        [-0.18410027, -0.45194484],
        [0.9281014, 0.53085498],
        [-0.14374509, 0.27370049],
        [-0.41635887, -0.38299653],
        [0.08711622, 0.93259929],
        [1.70580611, -0.11219234],
    ]
)
Y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
R_TOL = 1e-4

XX = np.array(
    [
        [0.11622591, -0.0317206],
        [0.77481731, 0.60935141],
        [1.25192108, -0.22367336],
        [0.53366841, -0.30312976],
        [1.52091956, -0.49283504],
        [-0.28162401, -2.10400981],
        [0.83680821, 1.72827342],
        [0.3084254, 0.33299982],
        [0.70472253, -0.73309052],
        [0.28893132, -0.38761769],
        [1.15514042, 0.0129463],
        [0.88407872, 0.35454207],
    ]
)
YY = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1])


def test_sample_regular():
    smote = SMOTE(random_state=RND_SEED)
    X_resampled, y_resampled = smote.fit_resample(X, Y)
    X_gt = np.array(
        [
            [0.11622591, -0.0317206],
            [0.77481731, 0.60935141],
            [1.25192108, -0.22367336],
            [0.53366841, -0.30312976],
            [1.52091956, -0.49283504],
            [-0.28162401, -2.10400981],
            [0.83680821, 1.72827342],
            [0.3084254, 0.33299982],
            [0.70472253, -0.73309052],
            [0.28893132, -0.38761769],
            [1.15514042, 0.0129463],
            [0.88407872, 0.35454207],
            [1.31301027, -0.92648734],
            [-1.11515198, -0.93689695],
            [-0.18410027, -0.45194484],
            [0.9281014, 0.53085498],
            [-0.14374509, 0.27370049],
            [-0.41635887, -0.38299653],
            [0.08711622, 0.93259929],
            [1.70580611, -0.11219234],
            [0.29307743, -0.14670439],
            [0.84976473, -0.15570176],
            [0.61319159, -0.11571668],
            [0.66052536, -0.28246517],
        ]
    )
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    )
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_regular_half():
    sampling_strategy = {0: 9, 1: 12}
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RND_SEED)
    X_resampled, y_resampled = smote.fit_resample(X, Y)
    X_gt = np.array(
        [
            [0.11622591, -0.0317206],
            [0.77481731, 0.60935141],
            [1.25192108, -0.22367336],
            [0.53366841, -0.30312976],
            [1.52091956, -0.49283504],
            [-0.28162401, -2.10400981],
            [0.83680821, 1.72827342],
            [0.3084254, 0.33299982],
            [0.70472253, -0.73309052],
            [0.28893132, -0.38761769],
            [1.15514042, 0.0129463],
            [0.88407872, 0.35454207],
            [1.31301027, -0.92648734],
            [-1.11515198, -0.93689695],
            [-0.18410027, -0.45194484],
            [0.9281014, 0.53085498],
            [-0.14374509, 0.27370049],
            [-0.41635887, -0.38299653],
            [0.08711622, 0.93259929],
            [1.70580611, -0.11219234],
            [0.36784496, -0.1953161],
        ]
    )
    y_gt = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_regular_with_nn():
    nn_k = NearestNeighbors(n_neighbors=6)
    smote = SMOTE(random_state=RND_SEED, k_neighbors=nn_k)
    X_resampled, y_resampled = smote.fit_resample(X, Y)
    X_gt = np.array(
        [
            [0.11622591, -0.0317206],
            [0.77481731, 0.60935141],
            [1.25192108, -0.22367336],
            [0.53366841, -0.30312976],
            [1.52091956, -0.49283504],
            [-0.28162401, -2.10400981],
            [0.83680821, 1.72827342],
            [0.3084254, 0.33299982],
            [0.70472253, -0.73309052],
            [0.28893132, -0.38761769],
            [1.15514042, 0.0129463],
            [0.88407872, 0.35454207],
            [1.31301027, -0.92648734],
            [-1.11515198, -0.93689695],
            [-0.18410027, -0.45194484],
            [0.9281014, 0.53085498],
            [-0.14374509, 0.27370049],
            [-0.41635887, -0.38299653],
            [0.08711622, 0.93259929],
            [1.70580611, -0.11219234],
            [0.29307743, -0.14670439],
            [0.84976473, -0.15570176],
            [0.61319159, -0.11571668],
            [0.66052536, -0.28246517],
        ]
    )
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    )
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_indices():
    smote = SMOTE(random_state=RND_SEED)
    smote.fit_resample(X, Y)
    indices = smote.get_sample_indices()
    indices_gt = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [4, 0],
            [5, 0],
            [6, 0],
            [7, 0],
            [8, 0],
            [9, 0],
            [10, 0],
            [11, 0],
            [12, 0],
            [13, 0],
            [14, 0],
            [15, 0],
            [16, 0],
            [17, 0],
            [18, 0],
            [19, 0],
            [0, 2],
            [0, 1],
            [0, 1],
            [7, 2],
        ]
    )
    assert_array_equal(indices, indices_gt)


def test_sample_indices_balanced_dataset():
    smote = SMOTE(random_state=RND_SEED)
    smote.fit_resample(XX, YY)
    indices = smote.get_sample_indices()
    indices_gt = np.array(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [3, 0],
            [4, 0],
            [5, 0],
            [6, 0],
            [7, 0],
            [8, 0],
            [9, 0],
            [10, 0],
            [11, 0],
        ]
    )
    assert_array_equal(indices, indices_gt)


def test_sample_indices_is_none():
    smote = SMOTE(random_state=RND_SEED)
    indices = smote.get_sample_indices()
    assert_array_equal(indices, None)


def test_smote_FutureWarning():
    smote = SMOTE(random_state=RND_SEED, n_jobs=1)
    with pytest.warns(FutureWarning) as record:
        smote.fit_resample(XX, YY)
    assert len(record) == 1
    assert (
        record[0].message.args[0]
        == "The parameter `n_jobs` has been deprecated in 0.10"
        " and will be removed in 0.12. You can pass an nearest"
        " neighbors estimator where `n_jobs` is already set instead."
    )
