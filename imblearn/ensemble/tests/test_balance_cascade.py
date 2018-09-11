"""Test the module balance cascade."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numpy as np

import pytest
from pytest import raises

from sklearn.utils.testing import assert_array_equal
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

from imblearn.ensemble import BalanceCascade

RND_SEED = 0
X = np.array([[0.11622591, -0.0317206], [0.77481731, 0.60935141], [
    1.25192108, -0.22367336
], [0.53366841, -0.30312976], [1.52091956,
                               -0.49283504], [-0.28162401, -2.10400981],
              [0.83680821,
               1.72827342], [0.3084254, 0.33299982], [0.70472253, -0.73309052],
              [0.28893132, -0.38761769], [1.15514042, 0.0129463], [
                  0.88407872, 0.35454207
              ], [1.31301027, -0.92648734], [-1.11515198, -0.93689695], [
                  -0.18410027, -0.45194484
              ], [0.9281014, 0.53085498], [-0.14374509, 0.27370049], [
                  -0.41635887, -0.38299653
              ], [0.08711622, 0.93259929], [1.70580611, -0.11219234]])
Y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])


@pytest.mark.filterwarnings('ignore:Class BalanceCascade is deprecated')
def test_fit_resample_auto():
    sampling_strategy = 'auto'
    bc = BalanceCascade(
        sampling_strategy=sampling_strategy,
        random_state=RND_SEED,
        return_indices=True)
    X_resampled, y_resampled, idx_under = bc.fit_resample(X, Y)
    X_gt = np.array(
        [[[1.15514042, 0.0129463], [0.08711622, 0.93259929],
          [0.70472253,
           -0.73309052], [-0.14374509, 0.27370049], [0.83680821, 1.72827342], [
               -0.18410027, -0.45194484
           ], [-0.28162401, -2.10400981], [-1.11515198, -0.93689695], [
               0.11622591, -0.0317206
           ], [1.25192108, -0.22367336], [0.53366841, -0.30312976], [
               1.52091956, -0.49283504
           ], [0.88407872, 0.35454207], [1.31301027, -0.92648734],
          [-0.41635887, -0.38299653], [1.70580611, -0.11219234]],
         [[0.28893132,
           -0.38761769], [0.83680821, 1.72827342], [0.3084254, 0.33299982],
          [0.70472253, -0.73309052], [-0.14374509, 0.27370049], [
              0.77481731, 0.60935141
          ], [-0.18410027, -0.45194484], [1.15514042, 0.0129463],
          [0.11622591, -0.0317206], [1.25192108, -0.22367336], [
              0.53366841, -0.30312976
          ], [1.52091956, -0.49283504], [0.88407872, 0.35454207], [
              1.31301027, -0.92648734
          ], [-0.41635887, -0.38299653], [1.70580611, -0.11219234]]])
    y_gt = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                     [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    idx_gt = np.array(
        [[10, 18, 8, 16, 6, 14, 5, 13, 0, 2, 3, 4, 11, 12, 17, 19],
         [9, 6, 7, 8, 16, 1, 14, 10, 0, 2, 3, 4, 11, 12, 17, 19]])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
    assert_array_equal(idx_under, idx_gt)


@pytest.mark.filterwarnings('ignore:Class BalanceCascade is deprecated')
def test_fit_resample_half():
    sampling_strategy = {0: 8, 1: 10}
    bc = BalanceCascade(
        sampling_strategy=sampling_strategy, random_state=RND_SEED)
    X_resampled, y_resampled = bc.fit_resample(X, Y)
    X_gt = np.array([[[-0.41635887, -0.38299653], [0.53366841, -0.30312976], [
        1.25192108, -0.22367336
    ], [1.70580611, -0.11219234], [1.52091956, -0.49283504], [
        0.11622591, -0.0317206
    ], [1.31301027, -0.92648734], [0.88407872, 0.35454207], [
        0.3084254, 0.33299982
    ], [0.08711622, 0.93259929], [-0.28162401, -2.10400981], [
        -0.14374509, 0.27370049
    ], [0.9281014, 0.53085498], [-0.18410027, -0.45194484],
                      [0.77481731, 0.60935141], [1.15514042, 0.0129463],
                      [-1.11515198, -0.93689695], [0.70472253, -0.73309052]]])
    y_gt = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


@pytest.mark.filterwarnings('ignore:Class BalanceCascade is deprecated')
def test_fit_resample_auto_early_stop():
    sampling_strategy = 'auto'
    estimator = LinearSVC(random_state=RND_SEED)
    bc = BalanceCascade(
        sampling_strategy=sampling_strategy,
        random_state=RND_SEED,
        return_indices=False,
        estimator=estimator,
        n_max_subset=1)
    X_resampled, y_resampled = bc.fit_resample(X, Y)
    X_gt = np.array([[[1.15514042, 0.0129463], [0.08711622, 0.93259929], [
        0.70472253, -0.73309052
    ], [-0.14374509, 0.27370049], [0.83680821, 1.72827342], [
        -0.18410027, -0.45194484
    ], [-0.28162401, -2.10400981], [-1.11515198, -0.93689695], [
        0.11622591, -0.0317206
    ], [1.25192108, -0.22367336], [0.53366841, -0.30312976], [
        1.52091956, -0.49283504
    ], [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234]]])
    y_gt = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


@pytest.mark.filterwarnings('ignore:Class BalanceCascade is deprecated')
def test_give_classifier_obj():
    sampling_strategy = 'auto'
    estimator = RandomForestClassifier(n_estimators=10, random_state=RND_SEED)
    bc = BalanceCascade(
        sampling_strategy=sampling_strategy,
        random_state=RND_SEED,
        return_indices=False,
        estimator=estimator)
    X_resampled, y_resampled = bc.fit_resample(X, Y)
    X_gt = np.array([[[1.15514042, 0.0129463], [0.08711622, 0.93259929], [
        0.70472253, -0.73309052
    ], [-0.14374509, 0.27370049], [0.83680821, 1.72827342], [
        -0.18410027, -0.45194484
    ], [-0.28162401, -2.10400981], [-1.11515198, -0.93689695], [
        0.11622591, -0.0317206
    ], [1.25192108, -0.22367336], [0.53366841, -0.30312976], [
        1.52091956, -0.49283504
    ], [0.88407872, 0.35454207], [1.31301027, -0.92648734],
                      [-0.41635887, -0.38299653], [1.70580611, -0.11219234]]])
    y_gt = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]])
    assert_array_equal(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


@pytest.mark.filterwarnings('ignore:Class BalanceCascade is deprecated')
def test_give_classifier_wrong_obj():
    sampling_strategy = 'auto'
    classifier = 2
    bc = BalanceCascade(
        sampling_strategy=sampling_strategy,
        random_state=RND_SEED,
        return_indices=True,
        estimator=classifier)
    with raises(ValueError, match="Invalid parameter `estimator`"):
        bc.fit_resample(X, Y)
