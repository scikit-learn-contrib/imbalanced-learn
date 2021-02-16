import pytest
import numpy as np

from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_array_equal

from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from imblearn.over_sampling import KMeansSMOTE
from imblearn.over_sampling import SMOTE


@pytest.fixture
def data():
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
    y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    return X, y


def test_kmeans_smote(data):
    X, y = data
    kmeans_smote = KMeansSMOTE(
        kmeans_estimator=1,
        random_state=42,
        cluster_balance_threshold=0.0,
        k_neighbors=5,
    )
    smote = SMOTE(random_state=42)

    X_res_1, y_res_1 = kmeans_smote.fit_resample(X, y)
    X_res_2, y_res_2 = smote.fit_resample(X, y)

    assert_allclose(X_res_1, X_res_2)
    assert_array_equal(y_res_1, y_res_2)

    assert kmeans_smote.nn_k_.n_neighbors == 6
    assert kmeans_smote.kmeans_estimator_.n_clusters == 1
    assert "batch_size" in kmeans_smote.kmeans_estimator_.get_params()


@pytest.mark.parametrize("k_neighbors", [2, NearestNeighbors(n_neighbors=3)])
@pytest.mark.parametrize(
    "kmeans_estimator",
    [
        3,
        KMeans(n_clusters=3, random_state=42),
        MiniBatchKMeans(n_clusters=3, random_state=42),
    ],
)
def test_sample_kmeans_custom(data, k_neighbors, kmeans_estimator):
    X, y = data
    kmeans_smote = KMeansSMOTE(
        random_state=42,
        kmeans_estimator=kmeans_estimator,
        k_neighbors=k_neighbors,
    )
    X_resampled, y_resampled = kmeans_smote.fit_resample(X, y)
    assert X_resampled.shape == (24, 2)
    assert y_resampled.shape == (24,)

    assert kmeans_smote.nn_k_.n_neighbors == 3
    assert kmeans_smote.kmeans_estimator_.n_clusters == 3


def test_sample_kmeans_not_enough_clusters():
    rng = np.random.RandomState(42)
    X = rng.randn(30, 2)
    y = np.array([1] * 20 + [0] * 10)

    smote = KMeansSMOTE(random_state=42, kmeans_estimator=30, k_neighbors=2)
    with pytest.raises(RuntimeError):
        smote.fit_resample(X, y)


@pytest.mark.parametrize("density_exponent", ["auto", 2])
@pytest.mark.parametrize("cluster_balance_threshold", ["auto", 0.8])
def test_sample_kmeans_density_estimation(
    data, density_exponent, cluster_balance_threshold
):
    X, y = data
    smote = KMeansSMOTE(
        random_state=42,
        density_exponent=density_exponent,
        cluster_balance_threshold=cluster_balance_threshold,
    )
    smote.fit_resample(X, y)


@pytest.mark.parametrize(
    "density_exponent, cluster_balance_threshold",
    [("xxx", "auto"), ("auto", "xxx")],
)
def test_kmeans_smote_param_error(data, density_exponent, cluster_balance_threshold):
    X, y = data
    kmeans_smote = KMeansSMOTE(
        density_exponent=density_exponent,
        cluster_balance_threshold=cluster_balance_threshold,
    )
    with pytest.raises(ValueError, match="should be 'auto' when a string"):
        kmeans_smote.fit_resample(X, y)
