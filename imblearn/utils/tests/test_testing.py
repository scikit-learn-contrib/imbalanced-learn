"""Test for the testing module"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import pytest

import numpy as np

from sklearn.neighbors._base import KNeighborsMixin

from imblearn.base import SamplerMixin
from imblearn.utils.testing import all_estimators, CustomNearestNeighbors

from imblearn.utils.testing import warns


def test_all_estimators():
    # check if the filtering is working with a list or a single string
    type_filter = "sampler"
    all_estimators(type_filter=type_filter)
    type_filter = ["sampler"]
    estimators = all_estimators(type_filter=type_filter)
    for estimator in estimators:
        # check that all estimators are sampler
        assert issubclass(estimator[1], SamplerMixin)

    # check that an error is raised when the type is unknown
    type_filter = "rnd"
    with pytest.raises(ValueError, match="Parameter type_filter must be 'sampler'"):
        all_estimators(type_filter=type_filter)


@pytest.mark.filterwarnings("ignore:The warns function is deprecated in 0.8")
def test_warns():
    import warnings

    with warns(UserWarning, match=r"must be \d+$"):
        warnings.warn("value must be 42", UserWarning)

    with pytest.raises(AssertionError, match="pattern not found"):
        with warns(UserWarning, match=r"must be \d+$"):
            warnings.warn("this is not here", UserWarning)

    with warns(UserWarning, match=r"aaa"):
        warnings.warn("cccccccccc", UserWarning)
        warnings.warn("bbbbbbbbbb", UserWarning)
        warnings.warn("aaaaaaaaaa", UserWarning)

    a, b, c = ("aaa", "bbbbbbbbbb", "cccccccccc")
    expected_msg = r"'{}' pattern not found in \['{}', '{}'\]".format(a, b, c)
    with pytest.raises(AssertionError, match=expected_msg):
        with warns(UserWarning, match=r"aaa"):
            warnings.warn("bbbbbbbbbb", UserWarning)
            warnings.warn("cccccccccc", UserWarning)


# TODO: remove in 0.9
def test_warns_deprecation():
    import warnings

    with pytest.warns(None) as record:
        with warns(UserWarning):
            warnings.warn("value must be 42")
    assert "The warns function is deprecated" in str(record[0].message)


def test_custom_nearest_neighbors():
    """Check that our custom nearest neighbors can be used for our internal
    duck-typing."""

    neareat_neighbors = CustomNearestNeighbors(n_neighbors=3)

    assert not isinstance(neareat_neighbors, KNeighborsMixin)
    assert hasattr(neareat_neighbors, "kneighbors")
    assert hasattr(neareat_neighbors, "kneighbors_graph")

    rng = np.random.RandomState(42)
    X = rng.randn(150, 3)
    y = rng.randint(0, 2, 150)
    neareat_neighbors.fit(X, y)

    distances, indices = neareat_neighbors.kneighbors(X)
    assert distances.shape == (150, 3)
    assert indices.shape == (150, 3)
    np.testing.assert_allclose(distances[:, 0], 0.0)
    np.testing.assert_allclose(indices[:, 0], np.arange(150))
