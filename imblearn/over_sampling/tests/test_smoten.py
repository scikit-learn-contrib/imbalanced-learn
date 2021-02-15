from collections import Counter

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
    X, y = data
    print(X, y)
    sampler = SMOTEN(random_state=0)
    X_res, y_res = sampler.fit_resample(X, y)
    print(X_res, y_res)
    print(Counter(y_res))
