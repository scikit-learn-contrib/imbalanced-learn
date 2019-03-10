import pytest
import numpy as np

from sklearn.utils.testing import assert_allclose
from sklearn.utils.testing import assert_array_equal
from sklearn.cluster import MiniBatchKMeans

from imblearn.over_sampling import (KMeansSMOTE, SMOTE)


@pytest.fixture
def data():
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
    y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    return X, y


def test_kmeans_smote(data):
    X, y = data
    kmeans_smote = KMeansSMOTE(kmeans_estimator=1,
                               random_state=42,
                               cluster_balance_threshold=0.0,
                               k_neighbors=5)
    smote = SMOTE(random_state=42)

    X_res_1, y_res_1 = kmeans_smote.fit_sample(X, y)
    X_res_2, y_res_2 = smote.fit_sample(X, y)

    assert_allclose(X_res_1, X_res_2)
    assert_array_equal(y_res_1, y_res_2)


def test_sample_kmeans_custom(data):
    X, y = data
    smote = KMeansSMOTE(random_state=42,
                        kmeans_estimator=MiniBatchKMeans(n_clusters=3, random_state=42),
                        k_neighbors=2)
    X_resampled, y_resampled = smote.fit_sample(X, y)
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
        [1.19141841, -0.82923193], [0.687674179, -0.3327227441],
        [1.24349671, -0.87451605], [0.3042074282, -0.093428711]
    ])

    y_gt = np.array([
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0
    ])

    assert_allclose(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_sample_kmeans_not_enough_clusters(data):
    np.random.seed(42)
    X = np.random.random((30, 2))
    y = np.array([1] * 20 + [0] * 10)

    smote = KMeansSMOTE(random_state=42,
                        kmeans_estimator=30,
                        k_neighbors=2)
    with pytest.raises(RuntimeError):
        smote.fit_sample(X, y)


def test_sample_kmeans(data):
    X, y = data
    smote = KMeansSMOTE(random_state=42,
                        kmeans_estimator=3,
                        k_neighbors=2)
    X_resampled, y_resampled = smote.fit_sample(X, y)
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
        [1.19141841, -0.82923193], [0.687674179, -0.3327227441],
        [1.24349671, -0.87451605], [0.3042074282, -0.093428711]
    ])

    y_gt = np.array([
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0
    ])

    assert_allclose(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)


def test_sample_kmeans_density_estimation(data):
    X, y = data
    smote = KMeansSMOTE(random_state=42,
                        kmeans_estimator=3,
                        k_neighbors=2,
                        density_exponent=2)
    X_resampled, y_resampled = smote.fit_sample(X, y)
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
        [1.19141841, -0.82923193], [0.687674179, -0.3327227441],
        [1.24349671, -0.87451605], [0.3042074282, -0.093428711]
    ])

    y_gt = np.array([
        0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0
    ])

    assert_allclose(X_resampled, X_gt)
    assert_array_equal(y_resampled, y_gt)
