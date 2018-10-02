"""Test the module random under sampler."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from collections import Counter

import pytest
import numpy as np
from sklearn.utils.testing import assert_array_equal

from imblearn.under_sampling import RandomUnderSampler

RND_SEED = 0
X = np.array([[0.04352327, -0.20515826], [0.92923648, 0.76103773], [
    0.20792588, 1.49407907
], [0.47104475, 0.44386323], [0.22950086, 0.33367433], [0.15490546, 0.3130677],
              [0.09125309, -0.85409574], [0.12372842, 0.6536186],
              [0.13347175, 0.12167502], [0.094035, -2.55298982]])
Y = np.array([1, 0, 1, 0, 1, 1, 1, 1, 0, 1])


def test_rus_fit_resample():
    rus = RandomUnderSampler(random_state=RND_SEED, replacement=True)
    X_resampled, y_resampled = rus.fit_resample(X, Y)

    X_gt = np.array([[0.92923648, 0.76103773], [0.47104475, 0.44386323],
                     [0.13347175, 0.12167502], [0.09125309, -0.85409574],
                     [0.12372842, 0.6536186], [0.04352327, -0.20515826]])
    y_gt = np.array([0, 0, 0, 1, 1, 1])

    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


@pytest.mark.filterwarnings("ignore:'return_indices' is deprecated from 0.4")
def test_rus_fit_resample_with_indices():
    rus = RandomUnderSampler(
        return_indices=True, random_state=RND_SEED, replacement=True)
    X_resampled, y_resampled, idx_under = rus.fit_resample(X, Y)

    X_gt = np.array([[0.92923648, 0.76103773], [0.47104475, 0.44386323],
                     [0.13347175, 0.12167502], [0.09125309, -0.85409574],
                     [0.12372842, 0.6536186], [0.04352327, -0.20515826]])
    y_gt = np.array([0, 0, 0, 1, 1, 1])
    idx_gt = np.array([1, 3, 8, 6, 7, 0])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


def test_rus_fit_resample_half():
    sampling_strategy = {0: 3, 1: 6}
    rus = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=RND_SEED,
        replacement=True)
    X_resampled, y_resampled = rus.fit_resample(X, Y)

    X_gt = np.array([[0.92923648, 0.76103773], [0.47104475, 0.44386323], [
        0.92923648, 0.76103773
    ], [0.15490546, 0.3130677], [0.15490546, 0.3130677],
                     [0.15490546, 0.3130677], [0.20792588, 1.49407907],
                     [0.15490546, 0.3130677], [0.12372842, 0.6536186]])
    y_gt = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


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
    X_hetero = np.array([['xxx', 1, 1.0], ['yyy', 2, 2.0], ['zzz', 3, 3.0]],
                        dtype=np.object)
    y = np.array([0, 0, 1])
    rus = RandomUnderSampler(random_state=RND_SEED)
    X_res, y_res = rus.fit_resample(X_hetero, y)

    assert X_res.shape[0] == 2
    assert y_res.shape[0] == 2
    assert X_res.dtype == object
