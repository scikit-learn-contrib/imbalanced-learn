import numpy as np
import pytest

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
