"""Test the module SMOTENC."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
#          Dzianis Dudnik
# License: MIT

from collections import Counter

import pytest

import numpy as np
from scipy import sparse

from sklearn.datasets import make_classification
from sklearn.utils.testing import assert_allclose

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


def data_heterogneous_masked():
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
    return X, y, [True, False, True]


def data_heterogneous_unordered_multiclass():
    rng = np.random.RandomState(42)
    X = np.empty((50, 4), dtype=object)
    # create 2 random continuous feature
    X[:, [1, 2]] = rng.randn(50, 2)
    # create a categorical feature using some string
    X[:, 0] = rng.choice(['a', 'b', 'c'], size=50).astype(object)
    # create a categorical feature using some integer
    X[:, 3] = rng.randint(3, size=50)
    y = np.array([0] * 10 + [1] * 15 + [2] * 25)
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


def test_smotenc_error():
    X, y, _ = data_heterogneous_unordered()
    categorical_features = [0, 10]
    smote = SMOTENC(random_state=0, categorical_features=categorical_features)
    with pytest.raises(ValueError, match="indices are out of range"):
        smote.fit_resample(X, y)


@pytest.mark.parametrize(
    "data",
    [data_heterogneous_ordered(), data_heterogneous_unordered(),
     data_heterogneous_masked(),
     data_sparse('csr'), data_sparse('csc')]
)
def test_smotenc(data):
    X, y, categorical_features = data
    smote = SMOTENC(random_state=0, categorical_features=categorical_features)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    assert X_resampled.dtype == X.dtype

    categorical_features = np.array(categorical_features)
    if categorical_features.dtype == bool:
        categorical_features = np.flatnonzero(categorical_features)
    for cat_idx in categorical_features:
        if sparse.issparse(X):
            assert set(X[:, cat_idx].data) == set(X_resampled[:, cat_idx].data)
            assert X[:, cat_idx].dtype == X_resampled[:, cat_idx].dtype
        else:
            assert set(X[:, cat_idx]) == set(X_resampled[:, cat_idx])
            assert X[:, cat_idx].dtype == X_resampled[:, cat_idx].dtype


# part of the common test which apply to SMOTE-NC even if it is not default
# constructible
def test_smotenc_check_target_type():
    X, _, categorical_features = data_heterogneous_unordered()
    y = np.linspace(0, 1, 30)
    smote = SMOTENC(categorical_features=categorical_features,
                    random_state=0)
    with pytest.warns(UserWarning, match='should be of types'):
        smote.fit_resample(X, y)


def test_smotenc_samplers_one_label():
    X, _, categorical_features = data_heterogneous_unordered()
    y = np.zeros(30)
    smote = SMOTENC(categorical_features=categorical_features,
                    random_state=0)
    with pytest.raises(ValueError, match='needs to have more than 1 class'):
        smote.fit(X, y)


def test_smotenc_fit():
    X, y, categorical_features = data_heterogneous_unordered()
    smote = SMOTENC(categorical_features=categorical_features,
                    random_state=0)
    smote.fit_resample(X, y)
    assert hasattr(smote, 'sampling_strategy_'), \
        "No fitted attribute sampling_strategy_"


def test_smotenc_fit_resample():
    X, y, categorical_features = data_heterogneous_unordered()
    target_stats = Counter(y)
    smote = SMOTENC(categorical_features=categorical_features,
                    random_state=0)
    X_res, y_res = smote.fit_resample(X, y)
    target_stats_res = Counter(y_res)
    n_samples = max(target_stats.values())
    assert all(value >= n_samples for value in Counter(y_res).values())


def test_smotenc_fit_resample_sampling_strategy():
    X, y, categorical_features = data_heterogneous_unordered_multiclass()
    expected_stat = Counter(y)[1]
    smote = SMOTENC(categorical_features=categorical_features,
                    random_state=0)
    sampling_strategy = {2: 25, 0: 25}
    smote.set_params(sampling_strategy=sampling_strategy)
    X_res, y_res = smote.fit_resample(X, y)
    assert Counter(y_res)[1] == expected_stat


def test_smotenc_pandas():
    pd = pytest.importorskip("pandas")
    # Check that the samplers handle pandas dataframe and pandas series
    X, y, categorical_features = data_heterogneous_unordered_multiclass()
    X_pd = pd.DataFrame(X)
    smote = SMOTENC(categorical_features=categorical_features,
                    random_state=0)
    X_res_pd, y_res_pd = smote.fit_resample(X_pd, y)
    X_res, y_res = smote.fit_resample(X, y)
    assert X_res_pd.tolist() == X_res.tolist()
    assert_allclose(y_res_pd, y_res)


def test_smotenc_preserve_dtype():
    X, y = make_classification(n_samples=50, n_classes=3, n_informative=4,
                               weights=[0.2, 0.3, 0.5], random_state=0)
    # Cast X and y to not default dtype
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    smote = SMOTENC(categorical_features=[1], random_state=0)
    X_res, y_res = smote.fit_resample(X, y)
    assert X.dtype == X_res.dtype, "X dtype is not preserved"
    assert y.dtype == y_res.dtype, "y dtype is not preserved"
