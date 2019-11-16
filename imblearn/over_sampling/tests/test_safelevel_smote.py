import pytest
import numpy as np
from collections import Counter

from sklearn.neighbors import NearestNeighbors
from scipy import sparse

from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_array_equal

from imblearn.over_sampling import SafeLevelSMOTE


def data_np():
    rng = np.random.RandomState(42)
    X = rng.randn(20, 2)
    y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    return X, y


def data_sparse(format):
    X = sparse.random(20, 2, density=0.3, format=format, random_state=42)
    y = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    return X, y


@pytest.mark.parametrize(
    "data",
    [data_np(), data_sparse('csr'), data_sparse('csc')]
)
def test_safelevel_smote(data):
    y_gt = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
                     0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    X, y = data
    safelevel_smote = SafeLevelSMOTE(random_state=42)
    X_res, y_res = safelevel_smote.fit_resample(X, y)

    assert X_res.shape == (24, 2)
    assert_array_equal(y_res, y_gt)


def test_sl_smote_nn():
    X, y = data_np()
    safelevel_smote = SafeLevelSMOTE(random_state=42)
    safelevel_smote_nn = SafeLevelSMOTE(
        random_state=42,
        k_neighbors=NearestNeighbors(n_neighbors=6),
        m_neighbors=NearestNeighbors(n_neighbors=11),
    )

    X_res_1, y_res_1 = safelevel_smote.fit_resample(X, y)
    X_res_2, y_res_2 = safelevel_smote_nn.fit_resample(X, y)

    assert_allclose(X_res_1, X_res_2)
    assert_array_equal(y_res_1, y_res_2)


def test_sl_smote_pd():
    pd = pytest.importorskip("pandas")
    X, y = data_np()
    X_pd = pd.DataFrame(X)
    safelevel_smote = SafeLevelSMOTE(random_state=42)
    X_res, y_res = safelevel_smote.fit_resample(X, y)
    X_res_pd, y_res_pd = safelevel_smote.fit_resample(X_pd, y)

    assert X_res_pd.tolist() == X_res.tolist()
    assert_allclose(y_res_pd, y_res)


def test_sl_smote_multiclass():
    rng = np.random.RandomState(42)
    X = rng.randn(50, 2)
    y = np.array([0] * 10 + [1] * 15 + [2] * 25)
    safelevel_smote = SafeLevelSMOTE(random_state=42)
    X_res, y_res = safelevel_smote.fit_resample(X, y)

    count_y_res = Counter(y_res)
    assert count_y_res[0] == 25
    assert count_y_res[1] == 25
    assert count_y_res[2] == 25
