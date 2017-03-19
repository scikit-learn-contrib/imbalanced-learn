"""Test the module random under sampler."""
from __future__ import print_function

from collections import Counter

import numpy as np
from numpy.testing import assert_array_equal, assert_equal

from imblearn.under_sampling import RandomUnderSampler

# Generate a global dataset to use
RND_SEED = 0
# Data generated for the toy example
X = np.array([[0.04352327, -0.20515826], [0.92923648, 0.76103773],
              [0.20792588, 1.49407907], [0.47104475, 0.44386323],
              [0.22950086, 0.33367433], [0.15490546, 0.3130677],
              [0.09125309, -0.85409574], [0.12372842, 0.6536186],
              [0.13347175, 0.12167502], [0.094035, -2.55298982]])
Y = np.array([1, 0, 1, 0, 1, 1, 1, 1, 0, 1])


def test_rus_init():
    # Define a ratio
    ratio = 'auto'
    rus = RandomUnderSampler(ratio=ratio, random_state=RND_SEED)

    assert_equal(rus.random_state, RND_SEED)


def test_rus_fit_sample():
    # Resample the data
    rus = RandomUnderSampler(random_state=RND_SEED)
    X_resampled, y_resampled = rus.fit_sample(X, Y)

    X_gt = np.array([[0.92923648, 0.76103773], [0.47104475, 0.44386323],
                     [0.13347175, 0.12167502], [0.09125309, -0.85409574],
                     [0.12372842, 0.6536186], [0.04352327, -0.20515826]])
    y_gt = np.array([0, 0, 0, 1, 1, 1])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_rus_fit_sample_with_indices():
    # Resample the data
    rus = RandomUnderSampler(return_indices=True, random_state=RND_SEED)
    X_resampled, y_resampled, idx_under = rus.fit_sample(X, Y)

    X_gt = np.array([[0.92923648, 0.76103773], [0.47104475, 0.44386323],
                     [0.13347175, 0.12167502], [0.09125309, -0.85409574],
                     [0.12372842, 0.6536186], [0.04352327, -0.20515826]])
    y_gt = np.array([0, 0, 0, 1, 1, 1])
    idx_gt = np.array([1, 3, 8, 6, 7, 0])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


def test_rus_fit_sample_half():
    # Resample the data
    ratio = 0.5
    rus = RandomUnderSampler(ratio=ratio, random_state=RND_SEED)
    X_resampled, y_resampled = rus.fit_sample(X, Y)

    X_gt = np.array([[0.92923648, 0.76103773], [0.47104475, 0.44386323],
                     [0.13347175, 0.12167502], [0.09125309, -0.85409574],
                     [0.12372842, 0.6536186], [0.04352327, -0.20515826],
                     [0.15490546, 0.3130677], [0.15490546, 0.3130677],
                     [0.15490546, 0.3130677]])
    y_gt = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_multiclass_fit_sample():
    # Make y to be multiclass
    y = Y.copy()
    y[5] = 2
    y[6] = 2

    # Resample the data
    rus = RandomUnderSampler(random_state=RND_SEED)
    X_resampled, y_resampled = rus.fit_sample(X, y)

    # Check the size of y
    count_y_res = Counter(y_resampled)
    assert_equal(count_y_res[0], 2)
    assert_equal(count_y_res[1], 2)
    assert_equal(count_y_res[2], 2)
