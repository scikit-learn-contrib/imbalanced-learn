"""Test the module SMOTE."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import print_function

import numpy as np
import pytest

from sklearn.utils.testing import assert_allclose, assert_array_equal
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.over_sampling import KMeansSMOTE

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


def test_smote_wrong_kind():
    kind = 'rnd'
    smote = SMOTE(kind=kind, random_state=RND_SEED)
    with pytest.raises(ValueError, match="Unknown kind for SMOTE"):
        smote.fit_sample(X, Y)


def test_sample_regular():
    smote = SMOTE(random_state=RND_SEED)
    X_resampled, y_resampled = smote.fit_sample(X, Y)
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
    X_resampled, y_resampled = smote.fit_sample(X, Y)
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
    X_resampled, y_resampled = smote.fit_sample(X, Y)
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
    X_resampled, y_resampled = smote.fit_sample(X, Y)
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
    X_resampled, y_resampled = smote.fit_sample(X, Y)
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
def test_fit_sample_nn_obj():
    kind = 'borderline1'
    nn_m = NearestNeighbors(n_neighbors=11)
    nn_k = NearestNeighbors(n_neighbors=6)
    smote = SMOTE(
        random_state=RND_SEED, kind=kind, k_neighbors=nn_k, m_neighbors=nn_m)
    X_resampled, y_resampled = smote.fit_sample(X, Y)
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
    X_resampled, y_resampled = smote.fit_sample(X, Y)
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
def test_wrong_nn():
    kind = 'borderline1'
    nn_m = 'rnd'
    nn_k = NearestNeighbors(n_neighbors=6)
    smote = SMOTE(
        random_state=RND_SEED, kind=kind, k_neighbors=nn_k, m_neighbors=nn_m)
    with pytest.raises(ValueError, match="has to be one of"):
        smote.fit_sample(X, Y)
    nn_k = 'rnd'
    nn_m = NearestNeighbors(n_neighbors=10)
    smote = SMOTE(
        random_state=RND_SEED, kind=kind, k_neighbors=nn_k, m_neighbors=nn_m)
    with pytest.raises(ValueError, match="has to be one of"):
        smote.fit_sample(X, Y)
    kind = 'regular'
    nn_k = 'rnd'
    smote = SMOTE(random_state=RND_SEED, kind=kind, k_neighbors=nn_k)
    with pytest.raises(ValueError, match="has to be one of"):
        smote.fit_sample(X, Y)


@pytest.mark.filterwarnings('ignore:"kind" is deprecated in 0.4 and will be')
@pytest.mark.filterwarnings('ignore:"svm_estimator" is deprecated in 0.4 and')
@pytest.mark.filterwarnings('ignore:"out_step" is deprecated in 0.4 and')
@pytest.mark.filterwarnings('ignore:"m_neighbors" is deprecated in 0.4 and')
def test_sample_with_nn_svm():
    kind = 'svm'
    nn_k = NearestNeighbors(n_neighbors=6)
    svm = SVC(random_state=RND_SEED)
    smote = SMOTE(
        random_state=RND_SEED, kind=kind, k_neighbors=nn_k, svm_estimator=svm)
    X_resampled, y_resampled = smote.fit_sample(X, Y)
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


@pytest.mark.filterwarnings('ignore:"kind" is deprecated in 0.4 and will be')
@pytest.mark.filterwarnings('ignore:"svm_estimator" is deprecated in 0.4 and')
def test_sample_regular_wrong_svm():
    kind = 'svm'
    nn_k = NearestNeighbors(n_neighbors=6)
    svm = 'rnd'
    smote = SMOTE(
        random_state=RND_SEED, kind=kind, k_neighbors=nn_k, svm_estimator=svm)

    with pytest.raises(ValueError, match="has to be one of"):
        smote.fit_sample(X, Y)


def test_borderline_smote_wrong_kind():
    bsmote = BorderlineSMOTE(kind='rand')
    with pytest.raises(ValueError, match='The possible "kind" of algorithm'):
        bsmote.fit_sample(X, Y)


@pytest.mark.parametrize('kind', ['borderline-1', 'borderline-2'])
def test_borderline_smote(kind):
    bsmote = BorderlineSMOTE(kind=kind, random_state=42)
    bsmote_nn = BorderlineSMOTE(kind=kind, random_state=42,
                                k_neighbors=NearestNeighbors(n_neighbors=6),
                                m_neighbors=NearestNeighbors(n_neighbors=11))

    X_res_1, y_res_1 = bsmote.fit_sample(X, Y)
    X_res_2, y_res_2 = bsmote_nn.fit_sample(X, Y)

    assert_allclose(X_res_1, X_res_2)
    assert_array_equal(y_res_1, y_res_2)


def test_svm_smote():
    svm_smote = SVMSMOTE(random_state=42)
    svm_smote_nn = SVMSMOTE(random_state=42,
                            k_neighbors=NearestNeighbors(n_neighbors=6),
                            m_neighbors=NearestNeighbors(n_neighbors=11),
                            svm_estimator=SVC(random_state=42))

    X_res_1, y_res_1 = svm_smote.fit_sample(X, Y)
    X_res_2, y_res_2 = svm_smote_nn.fit_sample(X, Y)

    assert_allclose(X_res_1, X_res_2)
    assert_array_equal(y_res_1, y_res_2)


def test_kmeans_smote():
    kmeans_smote = KMeansSMOTE(kmeans_estimator=1, random_state=42)
    smote = SMOTE(random_state=42)

    X_res_1, y_res_1 = kmeans_smote.fit_sample(X, Y)
    X_res_2, y_res_2 = smote.fit_sample(X, Y)

    assert_allclose(X_res_1, X_res_2)
    assert_array_equal(y_res_1, y_res_2)


def test_sample_kmeans():
    smote = KMeansSMOTE(random_state=RND_SEED,
                        kmeans_estimator=3,
                        k_neighbors=2)
    X_resampled, y_resampled = smote.fit_sample(X, Y)
    X_gt = np.array([
        [0.11622591, -0.0317206], [0.77481731, 0.60935141],
        [1.25192108, -0.22367336], [0.53366841, -0.30312976],
        [1.52091956, -0.49283504], [-0.28162401, -2.10400981],
        [0.83680821, 1.72827342], [0.3084254, 0.33299982],
        [0.70472253, -0.73309052], [0.28893132, -0.38761769],
        [1.15514042, 0.0129463], [0.88407872, 0.35454207],
        [1.31301027, -0.92648734], [-1.11515198, -0.93689695],
        [-0.18410027, -0.45194484], [0.9281014, 0.53085498],
        [-0.14374509, 0.27370049], [-0.41635887, -0.38299653],
        [0.08711622, 0.93259929], [1.70580611, -0.11219234],
        [0.98135505, 0.22510668], [0.80404478, -0.2732194],
        [0.91314969, -0.37604899], [0.82740979, -0.35957364]
    ])

    y_gt = np.array([
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0
    ])

    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)


def test_sample_kmeans_density_estimation():
    smote = KMeansSMOTE(random_state=RND_SEED,
                        kmeans_estimator=3,
                        k_neighbors=2,
                        density_exponent=2)
    X_resampled, y_resampled = smote.fit_sample(X, Y)
    X_gt = np.array([
        [0.11622591, -0.0317206], [0.77481731, 0.60935141],
        [1.25192108, -0.22367336], [0.53366841, -0.30312976],
        [1.52091956, -0.49283504], [-0.28162401, -2.10400981],
        [0.83680821, 1.72827342], [0.3084254, 0.33299982],
        [0.70472253, -0.73309052], [0.28893132, -0.38761769],
        [1.15514042, 0.0129463], [0.88407872, 0.35454207],
        [1.31301027, -0.92648734], [-1.11515198, -0.93689695],
        [-0.18410027, -0.45194484], [0.9281014, 0.53085498],
        [-0.14374509, 0.27370049], [-0.41635887, -0.38299653],
        [0.08711622, 0.93259929], [1.70580611, -0.11219234],
        [0.98135505, 0.22510668], [0.80404478, -0.2732194],
        [0.91314969, -0.37604899], [0.82740979, -0.35957364]
    ])

    y_gt = np.array([
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0
    ])

    assert_allclose(X_resampled, X_gt, rtol=R_TOL)
    assert_array_equal(y_resampled, y_gt)
