"""Test the module SMOTE."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numpy as np
import pytest

from sklearn.utils.testing import assert_allclose, assert_array_equal
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE


RND_SEED = 0
X = np.array([[0.11622591, -0.0317206], [0.77481731, 0.60935141],
              [1.25192108, -0.22367336], [0.53366841, -0.30312976],
              [1.52091956, -0.49283504], [-0.28162401, -2.10400981],
              [0.83680821, 1.72827342], [0.3084254, 0.33299982],
              [0.70472253, -0.73309052], [0.28893132, -0.38761769],
              [1.15514042, 0.0129463], [0.88407872, 0.35454207],
              [1.31301027, -0.92648734], [-1.11515198, -0.93689695],
              [-0.18410027, -0.45194484], [0.9281014, 0.53085498],
              [-0.14374509, 0.27370049], [-0.41635887, -0.38299653],
              [0.08711622, 0.93259929], [1.70580611, -0.11219234]])
Y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
R_TOL = 1e-4


def test_sample_regular():
    smote = SMOTE(random_state=RND_SEED)
    X_resampled, y_resampled = smote.fit_resample(X, Y)
    X_gt = np.array([[0.11622591, -0.0317206], [0.77481731, 0.60935141], [
        1.25192108, -0.22367336
    ], [0.53366841, -0.30312976], [1.52091956, -0.49283504], [
        -0.28162401, -2.10400981
    ], [0.83680821, 1.72827342], [0.3084254, 0.33299982], [
        0.70472253, -0.73309052
    ], [0.28893132, -0.38761769], [1.15514042, 0.0129463], [
        0.88407872, 0.35454207
    ], [1.31301027, -0.92648734], [-1.11515198, -0.93689695], [
        -0.18410027, -0.45194484
    ], [0.9281014, 0.53085498], [-0.14374509, 0.27370049], [
        -0.41635887, -0.38299653
    ], [0.08711622, 0.93259929], [1.70580611, -0.11219234],
                     [0.29307743, -0.14670439], [0.84976473, -0.15570176],
                     [0.61319159, -0.11571668], [0.66052536, -0.28246517]])
    y_gt = np.array([
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0
    ])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_regular_half():
    sampling_strategy = {0: 9, 1: 12}
    smote = SMOTE(
        sampling_strategy=sampling_strategy, random_state=RND_SEED)
    X_resampled, y_resampled = smote.fit_resample(X, Y)
    X_gt = np.array([[0.11622591, -0.0317206], [0.77481731, 0.60935141], [
        1.25192108, -0.22367336
    ], [0.53366841, -0.30312976], [1.52091956, -0.49283504], [
        -0.28162401, -2.10400981
    ], [0.83680821, 1.72827342], [0.3084254, 0.33299982], [
        0.70472253, -0.73309052
    ], [0.28893132, -0.38761769], [1.15514042, 0.0129463], [
        0.88407872, 0.35454207
    ], [1.31301027, -0.92648734], [-1.11515198, -0.93689695], [
        -0.18410027, -0.45194484
    ], [0.9281014, 0.53085498], [-0.14374509, 0.27370049],
                     [-0.41635887, -0.38299653], [0.08711622, 0.93259929],
                     [1.70580611, -0.11219234], [0.36784496, -0.1953161]])
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


@pytest.mark.filterwarnings('ignore:"kind" is deprecated in 0.4 and will be')
def test_sample_borderline1():
    kind = 'borderline1'
    smote = SMOTE(random_state=RND_SEED, kind=kind)
    X_resampled, y_resampled = smote.fit_resample(X, Y)
    X_gt = np.array([[0.11622591, -0.0317206], [0.77481731, 0.60935141], [
        1.25192108, -0.22367336
    ], [0.53366841, -0.30312976], [1.52091956, -0.49283504], [
        -0.28162401, -2.10400981
    ], [0.83680821, 1.72827342], [0.3084254, 0.33299982], [
        0.70472253, -0.73309052
    ], [0.28893132, -0.38761769], [1.15514042, 0.0129463], [
        0.88407872, 0.35454207
    ], [1.31301027, -0.92648734], [-1.11515198, -0.93689695], [
        -0.18410027, -0.45194484
    ], [0.9281014, 0.53085498], [-0.14374509, 0.27370049], [
        -0.41635887, -0.38299653
    ], [0.08711622, 0.93259929], [1.70580611, -0.11219234],
                     [0.3765279, -0.2009615], [0.55276636, -0.10550373],
                     [0.45413452, -0.08883319], [1.21118683, -0.22817957]])
    y_gt = np.array([
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0
    ])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


@pytest.mark.filterwarnings('ignore:"kind" is deprecated in 0.4 and will be')
def test_sample_borderline2():
    kind = 'borderline2'
    smote = SMOTE(random_state=RND_SEED, kind=kind)
    X_resampled, y_resampled = smote.fit_resample(X, Y)
    X_gt = np.array([[0.11622591, -0.0317206], [0.77481731, 0.60935141], [
        1.25192108, -0.22367336
    ], [0.53366841, -0.30312976], [1.52091956, -0.49283504], [
        -0.28162401, -2.10400981
    ], [0.83680821, 1.72827342], [0.3084254, 0.33299982], [
        0.70472253, -0.73309052
    ], [0.28893132, -0.38761769], [1.15514042, 0.0129463], [
        0.88407872, 0.35454207
    ], [1.31301027, -0.92648734], [-1.11515198, -0.93689695], [
        -0.18410027, -0.45194484
    ], [0.9281014, 0.53085498], [-0.14374509, 0.27370049],
                     [-0.41635887, -0.38299653], [0.08711622, 0.93259929],
                     [1.70580611, -0.11219234], [0.47436888, -0.2645749],
                     [1.07844561, -0.19435291], [0.33339622, 0.49870937]])
    y_gt = np.array(
        [0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


@pytest.mark.filterwarnings('ignore:"kind" is deprecated in 0.4 and will be')
@pytest.mark.filterwarnings('ignore:"svm_estimator" is deprecated in 0.4 and')
@pytest.mark.filterwarnings('ignore:"out_step" is deprecated in 0.4 and')
@pytest.mark.filterwarnings('ignore:"m_neighbors" is deprecated in 0.4 and')
def test_sample_svm():
    kind = 'svm'
    smote = SMOTE(random_state=RND_SEED, kind=kind)
    X_resampled, y_resampled = smote.fit_resample(X, Y)
    X_gt = np.array([[0.11622591, -0.0317206],
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
                     [0.47436887, -0.2645749],
                     [1.07844562, -0.19435291],
                     [1.44228238, -1.31256615],
                     [1.25636713, -1.04463226]])
    y_gt = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1,
                     1, 0, 1, 0, 0, 0, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


@pytest.mark.filterwarnings('ignore:"kind" is deprecated in 0.4 and will be')
@pytest.mark.filterwarnings('ignore:"m_neighbors" is deprecated in 0.4 and')
def test_fit_resample_nn_obj():
    kind = 'borderline1'
    nn_m = NearestNeighbors(n_neighbors=11)
    nn_k = NearestNeighbors(n_neighbors=6)
    smote = SMOTE(
        random_state=RND_SEED, kind=kind, k_neighbors=nn_k, m_neighbors=nn_m)
    X_resampled, y_resampled = smote.fit_resample(X, Y)
    X_gt = np.array([[0.11622591, -0.0317206], [0.77481731, 0.60935141], [
        1.25192108, -0.22367336
    ], [0.53366841, -0.30312976], [1.52091956, -0.49283504], [
        -0.28162401, -2.10400981
    ], [0.83680821, 1.72827342], [0.3084254, 0.33299982], [
        0.70472253, -0.73309052
    ], [0.28893132, -0.38761769], [1.15514042, 0.0129463], [
        0.88407872, 0.35454207
    ], [1.31301027, -0.92648734], [-1.11515198, -0.93689695], [
        -0.18410027, -0.45194484
    ], [0.9281014, 0.53085498], [-0.14374509, 0.27370049], [
        -0.41635887, -0.38299653
    ], [0.08711622, 0.93259929], [1.70580611, -0.11219234],
                     [0.3765279, -0.2009615], [0.55276636, -0.10550373],
                     [0.45413452, -0.08883319], [1.21118683, -0.22817957]])
    y_gt = np.array([
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0
    ])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_regular_with_nn():
    nn_k = NearestNeighbors(n_neighbors=6)
    smote = SMOTE(random_state=RND_SEED, k_neighbors=nn_k)
    X_resampled, y_resampled = smote.fit_resample(X, Y)
    X_gt = np.array([[0.11622591, -0.0317206], [0.77481731, 0.60935141], [
        1.25192108, -0.22367336
    ], [0.53366841, -0.30312976], [1.52091956, -0.49283504], [
        -0.28162401, -2.10400981
    ], [0.83680821, 1.72827342], [0.3084254, 0.33299982], [
        0.70472253, -0.73309052
    ], [0.28893132, -0.38761769], [1.15514042, 0.0129463], [
        0.88407872, 0.35454207
    ], [1.31301027, -0.92648734], [-1.11515198, -0.93689695], [
        -0.18410027, -0.45194484
    ], [0.9281014, 0.53085498], [-0.14374509, 0.27370049], [
        -0.41635887, -0.38299653
    ], [0.08711622, 0.93259929], [1.70580611, -0.11219234],
                     [0.29307743, -0.14670439], [0.84976473, -0.15570176],
                     [0.61319159, -0.11571668], [0.66052536, -0.28246517]])
    y_gt = np.array([
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0
    ])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


@pytest.mark.filterwarnings('ignore:"kind" is deprecated in 0.4 and will be')
@pytest.mark.filterwarnings('ignore:"m_neighbors" is deprecated in 0.4 and')
@pytest.mark.filterwarnings('ignore:"svm_estimator" is deprecated in 0.4 and')
@pytest.mark.parametrize(
    "smote_params, err_msg",
    [({"kind": "rnd"}, "Unknown kind for SMOTE"),
     ({"kind": "borderline1",
       "k_neighbors": NearestNeighbors(n_neighbors=6),
       "m_neighbors": 'rnd'}, "has to be one of"),
     ({"k_neighbors": 'rnd',
       "m_neighbors": NearestNeighbors(n_neighbors=10)}, "has to be one of"),
     ({"kind": "regular",
       "k_neighbors": 'rnd'}, "has to be one of"),
     ({"kind": "svm",
       "k_neighbors": NearestNeighbors(n_neighbors=6),
       "svm_estimator": 'rnd'}, "has to be one of")]
)
def test_smote_error_passing_estimator(smote_params, err_msg):
    smote = SMOTE(**smote_params)
    with pytest.raises(ValueError, match=err_msg):
        smote.fit_resample(X, Y)


@pytest.mark.filterwarnings('ignore:"kind" is deprecated in 0.4 and will be')
@pytest.mark.filterwarnings('ignore:"svm_estimator" is deprecated in 0.4 and')
@pytest.mark.filterwarnings('ignore:"out_step" is deprecated in 0.4 and')
@pytest.mark.filterwarnings('ignore:"m_neighbors" is deprecated in 0.4 and')
def test_sample_with_nn_svm():
    kind = 'svm'
    nn_k = NearestNeighbors(n_neighbors=6)
    svm = SVC(gamma='scale', random_state=RND_SEED)
    smote = SMOTE(
        random_state=RND_SEED, kind=kind, k_neighbors=nn_k, svm_estimator=svm)
    X_resampled, y_resampled = smote.fit_resample(X, Y)
    X_gt = np.array([[0.11622591, -0.0317206],
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
                     [0.47436887, -0.2645749],
                     [1.07844562, -0.19435291],
                     [1.44228238, -1.31256615],
                     [1.25636713, -1.04463226]])
    y_gt = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
                     1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)
