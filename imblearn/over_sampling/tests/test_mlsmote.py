"""Test the module MLSMOTE."""

import numpy as np
import pytest

from sklearn.datasets import make_multilabel_classification
from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import assert_array_equal

from imblearn.over_sampling import MLSMOTE

R_TOL = 1e-4


def data_heterogneous_ordered():
    rng = np.random.RandomState(42)
    X = np.empty((30, 4), dtype=object)
    # create 2 random continuous feature
    X[:, :2] = rng.randn(30, 2)
    # create a categorical feature using some string
    X[:, 2] = rng.choice(["a", "b", "c"], size=30).astype(object)
    # create a categorical feature using some integer
    X[:, 3] = rng.randint(3, size=30)
    y = [[0, 2, 3]] * 5 + [[1, 2, 3, 4]] * 2 + [[1, 2]] * 3 + [[1]] * 20
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
    y = [[0, 2, 3]] * 5 + [[1, 2, 3, 4]] * 2 + [[1, 2]] * 3 + [[1]] * 20
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
    y = [[0, 2, 3]] * 5 + [[1, 2, 3, 4]] * 2 + [[1, 2]] * 3 + [[1]] * 20
    # return the categories
    return X, y, [True, False, True]


def data_sparse():
    X, y = make_multilabel_classification(
        n_samples=20, n_features=5, return_indicator="sparse", random_state=42
    )
    return X, y, []


def data_dense():
    X, y = make_multilabel_classification(
        n_samples=20, n_features=5, return_indicator="dense", random_state=42
    )
    return X, y, []


def data_list_of_lists():
    X, y = make_multilabel_classification(
        n_samples=20, n_features=5, return_indicator=False, random_state=42
    )
    return X, y, []


def test_mlsmote_categorical_features_error():
    X, y, _ = data_heterogneous_unordered()
    categorical_features = [0, 10]
    smote = MLSMOTE(categorical_features=categorical_features)
    with pytest.raises(ValueError, match="indices are out of range"):
        smote.fit_resample(X, y)


def test_mlsmote_invalid_strategy_error():
    _, _, categorical_features = data_heterogneous_unordered()
    with pytest.raises(
        ValueError,
        match="Sampling Strategy can only be one of:",
    ):
        _ = MLSMOTE(categorical_features=categorical_features, sampling_strategy="foo")


@pytest.mark.parametrize(
    "data",
    [
        data_heterogneous_ordered(),
        data_heterogneous_unordered(),
        data_heterogneous_masked(),
        data_sparse(),
        data_dense(),
        data_list_of_lists(),
    ],
)
def test_mlsmote(data):
    X, y, categorical_features = data
    smote = MLSMOTE(categorical_features=categorical_features)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    assert X_resampled.dtype == X.dtype
    assert type(y) == type(y_resampled)

    categorical_features = np.array(categorical_features)
    if categorical_features.dtype == bool:
        categorical_features = np.flatnonzero(categorical_features)
    for cat_idx in categorical_features:
        assert set(X[:, cat_idx]) == set(X_resampled[:, cat_idx])
        assert X[:, cat_idx].dtype == X_resampled[:, cat_idx].dtype


def test_mlsmote_fit_resample_1():
    X, y, categorical_features = data_heterogneous_unordered()
    classes = set([a for x in y for a in x])
    smote = MLSMOTE(categorical_features=categorical_features)
    _, y_res = smote.fit_resample(X, y)
    classes_res = set([a for x in y_res for a in x])

    assert classes == classes_res
    assert hasattr(
        smote, "sampling_strategy_"
    ), "No fitted attribute sampling_strategy_"


def test_mlsmote_fit_resample_2():
    X = np.array(
        [
            [25.0, 34.0],
            [38.0, 10.0],
            [47.0, 7.0],
            [32.0, 15.0],
            [23.0, 27.0],
            [36.0, 9.0],
            [45.0, 10.0],
            [39.0, 7.0],
            [29.0, 26.0],
            [31.0, 18.0],
            [36.0, 6.0],
            [37.0, 7.0],
            [44.0, 10.0],
            [42.0, 16.0],
            [39.0, 5.0],
            [44.0, 9.0],
            [33.0, 13.0],
            [36.0, 12.0],
            [32.0, 6.0],
            [28.0, 9.0],
        ]
    )

    y = np.array(
        [
            [0, 0],
            [1, 1],
            [1, 0],
            [1, 1],
            [0, 0],
            [1, 1],
            [1, 1],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [0, 1],
            [1, 1],
            [0, 1],
        ]
    )

    X_resampled_exp = np.array(
        [
            [25.0, 34.0],
            [38.0, 10.0],
            [47.0, 7.0],
            [32.0, 15.0],
            [23.0, 27.0],
            [36.0, 9.0],
            [45.0, 10.0],
            [39.0, 7.0],
            [29.0, 26.0],
            [31.0, 18.0],
            [36.0, 6.0],
            [37.0, 7.0],
            [44.0, 10.0],
            [42.0, 16.0],
            [39.0, 5.0],
            [44.0, 9.0],
            [33.0, 13.0],
            [36.0, 12.0],
            [32.0, 6.0],
            [28.0, 9.0],
            [38.95071431, 6.34003029],
            [42.22519874, 6.10833449],
            [33.83699557, 12.99774833],
            [36.06175348, 5.12036059],
            [38.43013104, 10.0],
            [36.08297745, 6.69575776],
            [40.54443985, 9.70877086],
            [37.80041708, 5.18666265],
            [41.80182894, 9.45606998],
            [34.91230996, 10.05030734],
            [32.23225206, 6.60754485],
        ]
    )

    y_resampled_exp = np.array(
        [
            [0, 0],
            [1, 1],
            [1, 0],
            [1, 1],
            [0, 0],
            [1, 1],
            [1, 1],
            [0, 1],
            [0, 0],
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [0, 1],
            [1, 1],
            [0, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
            [1, 1],
        ]
    )

    smote = MLSMOTE(categorical_features=[], random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    print(X_resampled)
    print(y_resampled)
    assert_allclose(X_resampled, X_resampled_exp, rtol=R_TOL)
    assert_array_equal(y_resampled, y_resampled_exp)
