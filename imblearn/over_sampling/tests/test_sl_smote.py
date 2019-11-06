import pytest
import numpy as np

from sklearn.neighbors import NearestNeighbors
from scipy import sparse

from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_array_equal

from imblearn.over_sampling import SLSMOTE


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
def test_slsmote(data):
    y_gt = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0,
                     0, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    X, y = data
    slsmote = SLSMOTE(random_state=42)
    X_res, y_res = slsmote.fit_resample(X, y)

    assert X_res.shape == (24, 2)
    assert_array_equal(y_res, y_gt)


def test_slsmote_nn():
    X, y = data_np()
    slsmote = SLSMOTE(random_state=42)
    slsmote_nn = SLSMOTE(
        random_state=42,
        k_neighbors=NearestNeighbors(n_neighbors=6),
        m_neighbors=NearestNeighbors(n_neighbors=11),
    )

    X_res_1, y_res_1 = slsmote.fit_resample(X, y)
    X_res_2, y_res_2 = slsmote_nn.fit_resample(X, y)

    assert_allclose(X_res_1, X_res_2)
    assert_array_equal(y_res_1, y_res_2)


def test_slsmote_pd():
    pd = pytest.importorskip("pandas")
    X, y = data_np()
    X_pd = pd.DataFrame(X)
    slsmote = SLSMOTE(random_state=42)
    X_res, y_res = slsmote.fit_resample(X, y)
    X_res_pd, y_res_pd = slsmote.fit_resample(X_pd, y)

    assert X_res_pd.tolist() == X_res.tolist()
    assert_allclose(y_res_pd, y_res)
