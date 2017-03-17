from imblearn.utils import check_neighbors_object

from sklearn.neighbors.base import KNeighborsMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.testing import assert_equal, assert_raises_regex


def test_check_neighbors_object():
    name = 'n_neighbors'
    n_neighbors = 1
    estimator = check_neighbors_object(name, n_neighbors)
    assert issubclass(type(estimator), KNeighborsMixin)
    assert_equal(estimator.n_neighbors, 1)
    estimator = check_neighbors_object(name, n_neighbors, 1)
    assert issubclass(type(estimator), KNeighborsMixin)
    assert_equal(estimator.n_neighbors, 2)
    estimator = NearestNeighbors(n_neighbors)
    assert estimator is check_neighbors_object(name, estimator)
    n_neighbors = 'rnd'
    assert_raises_regex(ValueError, "has to be one of",
                        check_neighbors_object, name, n_neighbors)
