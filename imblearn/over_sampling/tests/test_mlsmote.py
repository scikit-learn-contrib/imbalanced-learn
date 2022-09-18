"""Test the module MLSMOTE."""


from collections import Counter

import pytest

import numpy as np
from scipy import sparse
from sklearn.preprocessing import MultiLabelBinarizer


from imblearn.over_sampling import MLSMOTE


def data_heterogneous_ordered():
    rng = np.random.RandomState(42)
    X = np.empty((30, 4), dtype=object)
    # create 2 random continuous feature
    X[:, :2] = rng.randn(30, 2)
    # create a categorical feature using some string
    X[:, 2] = rng.choice(["a", "b", "c"], size=30).astype(object)
    # create a categorical feature using some integer
    X[:, 3] = rng.randint(3, size=30)
    y = np.array([[0, 2, 3]] * 5 + [[1, 2, 3, 4]]*2 + [[1, 2]]*3+[[1]] * 20)
    # return the categories
    return X, y, [2, 3]


def data_heterogneous_unordered():
    rng = np.random.RandomState(42)
    X = np.empty((30, 4), dtype=object)
    # create 2 random continuous feature
    X[:, [1, 2]] = rng.randn(30, 2)
    # create a categorical feature using some string
    X[:, 0] = rng.choice(["a", "b", "c"], size=30).astype(object)
    # create a categorical feature using some integer
    X[:, 3] = rng.randint(3, size=30)
    y = np.array([[0, 2, 3]] * 5 + [[1, 2, 3, 4]]*2 + [[1, 2]]*3+[[1]] * 20)
    # return the categories
    return X, y, [0, 3]


def data_heterogneous_masked():
    rng = np.random.RandomState(42)
    X = np.empty((30, 4), dtype=object)
    # create 2 random continuous feature
    X[:, [1, 2]] = rng.randn(30, 2)
    # create a categorical feature using some string
    X[:, 0] = rng.choice(["a", "b", "c"], size=30).astype(object)
    # create a categorical feature using some integer
    X[:, 3] = rng.randint(3, size=30)
    y = np.array([[0, 2, 3]] * 5 + [[1, 2, 3, 4]]*2 + [[1, 2]]*3+[[1]] * 20)
    # return the categories
    return X, y, [True, False, True]


def data_sparse():
    rng = np.random.RandomState(42)
    X = np.empty((30, 4), dtype=np.float64)
    # create 2 random continuous feature
    X[:, [1, 2]] = rng.randn(30, 2)
    # create a categorical feature using some string
    X[:, 0] = rng.randint(3, size=30)
    # create a categorical feature using some integer
    X[:, 3] = rng.randint(3, size=30)
    y = np.array([[0, 2, 3]] * 5 + [[1, 2, 3, 4]]*2 + [[1, 2]]*3+[[1]] * 20)
    labelBinarizer = MultiLabelBinarizer()
    y = labelBinarizer.fit_transform(y)
    y = sparse.csr_matrix(y)
    return X, y, [0, 3]


def test_mlsmote_error():
    X, y, _ = data_heterogneous_unordered()
    categorical_features = [0, 10]
    smote = MLSMOTE(categorical_features=categorical_features)
    with pytest.raises(ValueError, match="indices are out of range"):
        smote.fit_resample(X, y)


@pytest.mark.parametrize(
    "data",
    [
        data_heterogneous_ordered(),
        data_heterogneous_unordered(),
        data_heterogneous_masked(),
        data_sparse()
    ],
)
def test_mlsmote(data):
    X, y, categorical_features = data
    smote = MLSMOTE(categorical_features=categorical_features)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    assert X_resampled.dtype == X.dtype

    categorical_features = np.array(categorical_features)
    if categorical_features.dtype == bool:
        categorical_features = np.flatnonzero(categorical_features)
    for cat_idx in categorical_features:
        assert set(X[:, cat_idx]) == set(X_resampled[:, cat_idx])
        assert X[:, cat_idx].dtype == X_resampled[:, cat_idx].dtype


def test_mlsmote_fit():
    X, y, categorical_features = data_heterogneous_unordered()
    smote = MLSMOTE(categorical_features=categorical_features)
    smote.fit_resample(X, y)
    assert hasattr(
        smote, "sampling_strategy_"
    ), "No fitted attribute sampling_strategy_"


def test_mlsmote_fit_resample():
    X, y, categorical_features = data_heterogneous_unordered()
    target_stats = Counter(np.unique(
        np.array([a for x in y for a in (x if isinstance(x, list) else [x])])))
    smote = MLSMOTE(categorical_features=categorical_features)
    _, y_res = smote.fit_resample(X, y)
    classes_res = np.unique(
        np.array([a for x in y_res
                  for a in (x if isinstance(x, list) else [x])]))
    _ = Counter(classes_res)
    n_samples = max(target_stats.values())
    assert all(value >= n_samples for value in Counter(classes_res).values())
