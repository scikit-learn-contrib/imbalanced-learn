"""Test the module under sampler."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import pytest
import numpy as np

from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_array_equal
from sklearn.neighbors import NearestNeighbors

from imblearn.over_sampling import ADASYN

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


def test_ada_init():
    sampling_strategy = "auto"
    ada = ADASYN(sampling_strategy=sampling_strategy, random_state=RND_SEED)
    assert ada.random_state == RND_SEED


def test_ada_fit_resample():
    ada = ADASYN(random_state=RND_SEED)
    X_resampled, y_resampled = ada.fit_resample(X, Y)
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
            [0.88161986, -0.2829741],
            [0.35681689, -0.18814597],
            [1.4148276, 0.05308106],
            [0.3136591, -0.31327875],
        ]
    )
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    )
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_ada_fit_resample_nn_obj():
    nn = NearestNeighbors(n_neighbors=6)
    ada = ADASYN(random_state=RND_SEED, n_neighbors=nn)
    X_resampled, y_resampled = ada.fit_resample(X, Y)
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
            [0.88161986, -0.2829741],
            [0.35681689, -0.18814597],
            [1.4148276, 0.05308106],
            [0.3136591, -0.31327875],
        ]
    )
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    )
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


@pytest.mark.parametrize(
    "adasyn_params, err_msg",
    [
        (
            {"sampling_strategy": {0: 9, 1: 12}},
            "No samples will be generated.",
        ),
        ({"n_neighbors": "rnd"}, "has to be one of"),
    ],
)
def test_adasyn_error(adasyn_params, err_msg):
    adasyn = ADASYN(**adasyn_params)
    with pytest.raises(ValueError, match=err_msg):
        adasyn.fit_resample(X, Y)
