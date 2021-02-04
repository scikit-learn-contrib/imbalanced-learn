"""Test the module SMOTE-RSB."""
# Authors: Zolisa Bleki <zolisa.bleki@gmail.com>
# License: MIT
import numpy as np
import pytest

from imblearn.over_sampling import SMOTERSB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_array_equal


@pytest.fixture
def data():
    X = np.array([[0.11622591, -0.0317206],
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
                  [1.70580611, -0.11219234]])
    y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    return X, y


def test_smote_rsb(data):
    X, y = data
    RND_SEED = 0
    R_TOL = 1e-4
    rsb = SMOTERSB(random_state=RND_SEED)
    X_res, y_res = rsb.fit_resample(X, y)
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
            [-0.09533627, -0.17126026],
            [1.45849179, -0.17293647],
            [0.8379596, -0.26946767],
            [0.38584956, -0.20702218],
        ]
    )

    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
         0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0]
    )
    assert_allclose(X_res, X_gt, rtol=R_TOL)
    assert_array_equal(y_res, y_gt)

    with pytest.raises(ValueError):
        SMOTERSB(similarity_threshold=1).fit_resample(X, y)

    with pytest.raises(TypeError):
        SMOTERSB(equivalence_set=(0, 1)).fit_resample(X, y)

    with pytest.raises(ValueError):
        SMOTERSB(equivalence_set=[0, 5]).fit_resample(X, y)

    with pytest.raises(TypeError):
        fake_callable = "func()"
        SMOTERSB(similarity_func=fake_callable).fit_resample(X, y)

    # different similarity matrix generates different values
    X_, y_ = SMOTERSB(similarity_func=cosine_similarity).fit_resample(X, y)
    assert y_.shape[0] == y_res.shape[0]
    assert not np.allclose(X_, X_res, rtol=R_TOL)
