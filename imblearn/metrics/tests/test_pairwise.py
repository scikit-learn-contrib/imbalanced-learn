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


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("k, r", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_value_difference_metric(data, dtype, k, r):
    # Check basic feature of the metric:
    # * the shape of the distance matrix is (n_samples, n_samples)
    # * computing pairwise distance of X is the same than explicitely between
    #   X and X.
    X, y = data

    encoder = OrdinalEncoder(dtype=dtype)
    classes = np.unique(y)

    X_encoded = encoder.fit_transform(X)

    vdm = ValueDifferenceMetric(classes, encoder.categories_, k=k, r=r)
    vdm.fit(X_encoded, y)

    dist_1 = vdm.pairwise(X_encoded)
    dist_2 = vdm.pairwise(X_encoded, X_encoded)

    np.testing.assert_allclose(dist_1, dist_2)
    assert dist_1.shape == (X.shape[0], X.shape[0])
    assert dist_2.shape == (X.shape[0], X.shape[0])


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("k, r", [(1, 1), (1, 2), (2, 1), (2, 2)])
def test_value_difference_metric_property(dtype, k, r):
    # Check the property of the vdm distance. Let's check the property
    # described in "Improved Heterogeneous Distance Functions", D.R. Wilson and
    # T.R. Martinez, Journal of Artificial Intelligence Research 6 (1997) 1-34
    # https://arxiv.org/pdf/cs/9701101.pdf
    #
    # "if an attribute color has three values red, green and blue, and the
    # application is to identify whether or not an object is an apple, red and
    # green would be considered closer thanred and blue because the former two
    # both have similar correlations with the output class apple."

    # defined our feature
    X = np.array(["green"] * 10 + ["red"] * 10 + ["blue"] * 10).reshape(-1, 1)
    # 0 - not an apple / 1 - an apple
    y = np.array([1] * 8 + [0] * 5 + [1] * 7 + [0] * 9 + [1], dtype=np.int32)

    encoder = OrdinalEncoder(dtype=dtype)
    classes = np.unique(y)

    X_encoded = encoder.fit_transform(X)
    vdm = ValueDifferenceMetric(classes, encoder.categories_, k=k, r=r)
    vdm.fit(X_encoded, y)

    sample_green = encoder.transform([["green"]])
    sample_red = encoder.transform([["red"]])
    sample_blue = encoder.transform([["blue"]])

    for sample in (sample_green, sample_red, sample_blue):
        # computing the distance between a sample of the same category should
        # give a null distance
        dist = vdm.pairwise(sample).squeeze()
        assert dist == pytest.approx(0)

    # check the property explained in the introduction example
    dist_1 = vdm.pairwise(sample_green, sample_red).squeeze()
    dist_2 = vdm.pairwise(sample_blue, sample_red).squeeze()
    dist_3 = vdm.pairwise(sample_blue, sample_green).squeeze()

    # green and red are very close
    # blue is closer to red than green
    assert dist_1 < dist_2
    assert dist_1 < dist_3
    assert dist_2 < dist_3