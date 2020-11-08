"""Test the module random under sampler."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from collections import Counter

import numpy as np
import pytest

from sklearn.utils._testing import assert_array_equal

from imblearn.under_sampling import RandomUnderSampler

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


@pytest.mark.parametrize(
    "sampling_strategy, expected_counts",
    [
        ("auto", {0: 3, 1: 3}),
        ({0: 3, 1: 6}, {0: 3, 1: 6}),
    ]
)
def test_rus_fit_resample(sampling_strategy, expected_counts):
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy)
    X_res, y_res = rus.fit_resample(X, Y)

    # check that there is not samples from class 0 resampled as class 1 and
    # vice-versa
    classes = [0, 1]
    for c0, c1 in (classes, classes[::-1]):
        X_c0 = X[Y == c0]
        X_c1 = X_res[y_res == c1]
        for s0 in X_c0:
            assert not np.isclose(s0, X_c1).all(axis=1).any()

    assert Counter(y_res) == expected_counts


def test_multiclass_fit_resample():
    y = Y.copy()
    y[5] = 2
    y[6] = 2
    rus = RandomUnderSampler(random_state=RND_SEED)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    count_y_res = Counter(y_resampled)
    assert count_y_res[0] == 2
    assert count_y_res[1] == 2
    assert count_y_res[2] == 2


def test_random_under_sampling_heterogeneous_data():
    X_hetero = np.array(
        [["xxx", 1, 1.0], ["yyy", 2, 2.0], ["zzz", 3, 3.0]], dtype=np.object
    )
    y = np.array([0, 0, 1])
    rus = RandomUnderSampler(random_state=RND_SEED)
    X_res, y_res = rus.fit_resample(X_hetero, y)

    assert X_res.shape[0] == 2
    assert y_res.shape[0] == 2
    assert X_res.dtype == object


def test_random_under_sampling_nan_inf():
    # check that we can undersample even with missing or infinite data
    # regression tests for #605
    rng = np.random.RandomState(42)
    n_not_finite = X.shape[0] // 3
    row_indices = rng.choice(np.arange(X.shape[0]), size=n_not_finite)
    col_indices = rng.randint(0, X.shape[1], size=n_not_finite)
    not_finite_values = rng.choice([np.nan, np.inf], size=n_not_finite)

    X_ = X.copy()
    X_[row_indices, col_indices] = not_finite_values

    rus = RandomUnderSampler(random_state=0)
    X_res, y_res = rus.fit_resample(X_, Y)

    assert y_res.shape == (6,)
    assert X_res.shape == (6, 2)
    assert np.any(~np.isfinite(X_res))
