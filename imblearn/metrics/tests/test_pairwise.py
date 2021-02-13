"""Test for the metrics that perform pairwise distance computation."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import numpy as np
import pytest

from sklearn.preprocessing import OrdinalEncoder

from imblearn.metrics.pairwise import ValueDifferenceMetric


@pytest.fixture
def data():
    rng = np.random.RandomState(0)

    feature_1 = ["A"] * 10 + ["B"] * 20 + ["C"] * 30
    feature_2 = ["A"] * 40 + ["B"] * 20
    feature_3 = ["A"] * 20 + ["B"] * 20 + ["C"] * 10 + ["D"] * 10
    X = np.array([feature_1, feature_2, feature_3], dtype=object).T
    rng.shuffle(X)
    y = rng.randint(low=0, high=3, size=X.shape[0])
    return X, y


def test_value_difference_metric(data):
    X, y = data

    encoder = OrdinalEncoder(dtype=np.int32)
    classes = np.unique(y)

    X_encoded = encoder.fit_transform(X)

    vdm = ValueDifferenceMetric(classes, encoder.categories_)
    vdm.fit(X_encoded, y)

    dist_1 = vdm.pairwise(X_encoded)
    dist_2 = vdm.pairwise(X_encoded, X_encoded)

    np.testing.assert_allclose(dist_1, dist_2)
    assert dist_1.shape == (X.shape[0], X.shape[0])
    assert dist_2.shape == (X.shape[0], X.shape[0])
