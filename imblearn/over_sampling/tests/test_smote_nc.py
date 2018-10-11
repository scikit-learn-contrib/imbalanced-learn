"""Test the module SMOTENC."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
#          Dzianis Dudnik
# License: MIT

import pytest

import numpy as np
from scipy import sparse

from imblearn.over_sampling import SMOTENC


def data_heterogneous_ordered():
    rng = np.random.RandomState(42)
    X = np.empty((30, 4), dtype=object)
    # create 2 random continuous feature
    X[:, :2] = rng.randn(30, 2)
    # create a categorical feature using some string
    X[:, 2] = rng.choice(['a', 'b', 'c'], size=30).astype(object)
    # create a categorical feature using some integer
    X[:, 3] = rng.randint(3, size=30)
    y = np.array([0] * 10 + [1] * 20)
    # return the categories
    return X, y, [2, 3]


def data_heterogneous_unordered():
    rng = np.random.RandomState(42)
    X = np.empty((30, 4), dtype=object)
    # create 2 random continuous feature
    X[:, [1, 2]] = rng.randn(30, 2)
    # create a categorical feature using some string
    X[:, 0] = rng.choice(['a', 'b', 'c'], size=30).astype(object)
    # create a categorical feature using some integer
    X[:, 3] = rng.randint(3, size=30)
    y = np.array([0] * 10 + [1] * 20)
    # return the categories
    return X, y, [0, 3]


def data_sparse(format):
    rng = np.random.RandomState(42)
    X = np.empty((30, 4), dtype=np.float64)
    # create 2 random continuous feature
    X[:, [1, 2]] = rng.randn(30, 2)
    # create a categorical feature using some string
    X[:, 0] = rng.randint(3, size=30)
    # create a categorical feature using some integer
    X[:, 3] = rng.randint(3, size=30)
    y = np.array([0] * 10 + [1] * 20)
    X = sparse.csr_matrix(X) if format == 'csr' else sparse.csc_matrix(X)
    return X, y, [0, 3]


@pytest.mark.parametrize(
    "data",
    [data_heterogneous_ordered(), data_heterogneous_unordered(),
     data_sparse('csr'), data_sparse('csc')]
)
def test_smote_nc(data):
    X, y, categorical_features = data
    smote = SMOTENC(random_state=0, categorical_features=categorical_features)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    assert X_resampled.dtype == X.dtype

    for cat_idx in categorical_features:
        if sparse.issparse(X):
            assert set(X[:, cat_idx].data) == set(X_resampled[:, cat_idx].data)
            assert X[:, cat_idx].dtype == X_resampled[:, cat_idx].dtype
        else:
            assert set(X[:, cat_idx]) == set(X_resampled[:, cat_idx])
            assert X[:, cat_idx].dtype == X_resampled[:, cat_idx].dtype
