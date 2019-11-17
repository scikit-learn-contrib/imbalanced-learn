"""Test the module under sampler."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from collections import Counter

import numpy as np
import pytest

from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_array_equal

from imblearn.over_sampling import RandomOverSampler

RND_SEED = 0
X = np.array(
    [
        [0.04352327, -0.20515826],
        [0.92923648, 0.76103773],
        [0.20792588, 1.49407907],
        [0.47104475, 0.44386323],
        [0.22950086, 0.33367433],
        [0.15490546, 0.3130677],
        [0.09125309, -0.85409574],
        [0.12372842, 0.6536186],
        [0.13347175, 0.12167502],
        [0.094035, -2.55298982],
    ]
)
Y = np.array([1, 0, 1, 0, 1, 1, 1, 1, 0, 1])


def test_ros_init():
    sampling_strategy = "auto"
    ros = RandomOverSampler(
        sampling_strategy=sampling_strategy, random_state=RND_SEED
    )
    assert ros.random_state == RND_SEED


@pytest.mark.parametrize("as_frame", [True, False], ids=['dataframe', 'array'])
def test_ros_fit_resample(as_frame):
    if as_frame:
        pd = pytest.importorskip("pandas")
        X_ = pd.DataFrame(X)
    else:
        X_ = X
    ros = RandomOverSampler(random_state=RND_SEED)
    X_resampled, y_resampled = ros.fit_resample(X_, Y)
    X_gt = np.array(
        [
            [0.04352327, -0.20515826],
            [0.92923648, 0.76103773],
            [0.20792588, 1.49407907],
            [0.47104475, 0.44386323],
            [0.22950086, 0.33367433],
            [0.15490546, 0.3130677],
            [0.09125309, -0.85409574],
            [0.12372842, 0.6536186],
            [0.13347175, 0.12167502],
            [0.094035, -2.55298982],
            [0.92923648, 0.76103773],
            [0.47104475, 0.44386323],
            [0.92923648, 0.76103773],
            [0.47104475, 0.44386323],
        ]
    )
    y_gt = np.array([1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0])

    if as_frame:
        assert hasattr(X_resampled, "loc")
        X_resampled = X_resampled.to_numpy()

    assert_allclose(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_ros_fit_resample_half():
    sampling_strategy = {0: 3, 1: 7}
    ros = RandomOverSampler(
        sampling_strategy=sampling_strategy, random_state=RND_SEED
    )
    X_resampled, y_resampled = ros.fit_resample(X, Y)
    X_gt = np.array(
        [
            [0.04352327, -0.20515826],
            [0.92923648, 0.76103773],
            [0.20792588, 1.49407907],
            [0.47104475, 0.44386323],
            [0.22950086, 0.33367433],
            [0.15490546, 0.3130677],
            [0.09125309, -0.85409574],
            [0.12372842, 0.6536186],
            [0.13347175, 0.12167502],
            [0.094035, -2.55298982],
        ]
    )
    y_gt = np.array([1, 0, 1, 0, 1, 1, 1, 1, 0, 1])
    assert_allclose(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_multiclass_fit_resample():
    y = Y.copy()
    y[5] = 2
    y[6] = 2
    ros = RandomOverSampler(random_state=RND_SEED)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    count_y_res = Counter(y_resampled)
    assert count_y_res[0] == 5
    assert count_y_res[1] == 5
    assert count_y_res[2] == 5


def test_random_over_sampling_heterogeneous_data():
    X_hetero = np.array(
        [["xxx", 1, 1.0], ["yyy", 2, 2.0], ["zzz", 3, 3.0]], dtype=np.object
    )
    y = np.array([0, 0, 1])
    ros = RandomOverSampler(random_state=RND_SEED)
    X_res, y_res = ros.fit_resample(X_hetero, y)

    assert X_res.shape[0] == 4
    assert y_res.shape[0] == 4
    assert X_res.dtype == object
    assert X_res[-1, 0] in X_hetero[:, 0]


def test_random_over_sampling_nan_inf():
    # check that we can oversample even with missing or infinite data
    # regression tests for #605
    rng = np.random.RandomState(42)
    n_not_finite = X.shape[0] // 3
    row_indices = rng.choice(np.arange(X.shape[0]), size=n_not_finite)
    col_indices = rng.randint(0, X.shape[1], size=n_not_finite)
    not_finite_values = rng.choice([np.nan, np.inf], size=n_not_finite)

    X_ = X.copy()
    X_[row_indices, col_indices] = not_finite_values

    ros = RandomOverSampler(random_state=0)
    X_res, y_res = ros.fit_resample(X_, Y)

    assert y_res.shape == (14,)
    assert X_res.shape == (14, 2)
    assert np.any(~np.isfinite(X_res))
