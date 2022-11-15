import numpy as np
import pytest
import warnings

from imblearn.over_sampling import SMOTEN


@pytest.fixture
def data():
    rng = np.random.RandomState(0)

    feature_1 = ["A"] * 10 + ["B"] * 20 + ["C"] * 30
    feature_2 = ["A"] * 40 + ["B"] * 20
    feature_3 = ["A"] * 20 + ["B"] * 20 + ["C"] * 10 + ["D"] * 10
    X = np.array([feature_1, feature_2, feature_3], dtype=object).T
    rng.shuffle(X)
    y = np.array([0] * 20 + [1] * 40, dtype=np.int32)
    y_labels = np.array(["not apple", "apple"], dtype=object)
    y = y_labels[y]
    return X, y


def test_smoten(data):
    # overall check for SMOTEN
    X, y = data
    sampler = SMOTEN(random_state=0)
    X_res, y_res = sampler.fit_resample(X, y)

    assert X_res.shape == (80, 3)
    assert y_res.shape == (80,)


def test_smoten_resampling():
    # check if the SMOTEN resample data as expected
    # we generate data such that "not apple" will be the minority class and
    # samples from this class will be generated. We will force the "blue"
    # category to be associated with this class. Therefore, the new generated
    # samples should as well be from the "blue" category.
    X = np.array(["green"] * 5 + ["red"] * 10 + ["blue"] * 7, dtype=object).reshape(
        -1, 1
    )
    y = np.array(
        ["apple"] * 5
        + ["not apple"] * 3
        + ["apple"] * 7
        + ["not apple"] * 5
        + ["apple"] * 2,
        dtype=object,
    )
    sampler = SMOTEN(random_state=0)
    X_res, y_res = sampler.fit_resample(X, y)

    X_generated, y_generated = X_res[X.shape[0] :], y_res[X.shape[0] :]
    np.testing.assert_array_equal(X_generated, "blue")
    np.testing.assert_array_equal(y_generated, "not apple")


def test_smoten_FutureWarning(data):
    # check that SMOTEN throws FutureWarning for "n_jobs" and "keepdims"
    X, y = data
    sampler = SMOTEN(random_state=0, n_jobs=0)
    with pytest.warns(FutureWarning) as record:
        sampler.fit_resample(X, y)
    assert len(record) == 2
    assert (
        record[0].message.args[0]
        == "The parameter `n_jobs` has been deprecated in 0.10"
        " and will be removed in 0.12. You can pass an nearest"
        " neighbors estimator where `n_jobs` is already set instead."
    )
    assert (
        record[1].message.args[0]
        == "Unlike other reduction functions (e.g. `skew`, `kurtosis`),"
        " the default behavior of `mode` typically preserves the axis it"
        " acts along. In SciPy 1.11.0, this behavior will change: the default"
        " value of `keepdims` will become False, the `axis` over which the "
        "statistic is taken will be eliminated, and the value None will no longer"
        " be accepted. Set `keepdims` to True or False to avoid this warning."
    )


@pytest.fixture
def data_balanced():
    rng = np.random.RandomState(0)

    feature_1 = ["A"] * 10 + ["B"] * 20 + ["C"] * 30
    feature_2 = ["A"] * 40 + ["B"] * 20
    feature_3 = ["A"] * 20 + ["B"] * 20 + ["C"] * 10 + ["D"] * 10
    X = np.array([feature_1, feature_2, feature_3], dtype=object).T
    rng.shuffle(X)
    y = np.array([0] * 30 + [1] * 30, dtype=np.int32)
    y_labels = np.array(["not apple", "apple"], dtype=object)
    y = y_labels[y]
    return X, y


def test_smoten_balanced_data(data_balanced):
    X, y = data_balanced
    sampler = SMOTEN(random_state=0)
    X_res, y_res = sampler.fit_resample(X, y)
    assert X_res.shape == (60, 3)
    assert y_res.shape == (60,)
