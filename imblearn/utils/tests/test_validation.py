"""Test for the validation helper"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from collections import Counter

import numpy as np
import pytest
from pytest import raises

from sklearn.neighbors.base import KNeighborsMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_random_state
from sklearn.externals import joblib
from sklearn.utils.testing import assert_array_equal

from imblearn.utils.testing import warns
from imblearn.utils import check_neighbors_object
from imblearn.utils import check_ratio
from imblearn.utils import hash_X_y
from imblearn.utils import check_target_type


def test_check_neighbors_object():
    name = 'n_neighbors'
    n_neighbors = 1
    estimator = check_neighbors_object(name, n_neighbors)
    assert issubclass(type(estimator), KNeighborsMixin)
    assert estimator.n_neighbors == 1
    estimator = check_neighbors_object(name, n_neighbors, 1)
    assert issubclass(type(estimator), KNeighborsMixin)
    assert estimator.n_neighbors == 2
    estimator = NearestNeighbors(n_neighbors)
    assert estimator is check_neighbors_object(name, estimator)
    n_neighbors = 'rnd'
    with raises(ValueError, match="has to be one of"):
        check_neighbors_object(name, n_neighbors)


@pytest.mark.parametrize(
    "target, output_target",
    [(np.array([0, 1, 1]), np.array([0, 1, 1])),
     (np.array([0, 1, 2]), np.array([0, 1, 2])),
     (np.array([[0, 1], [1, 0]]), np.array([1, 0]))]
)
def test_check_target_type(target, output_target):
    converted_target = check_target_type(target.astype(int))
    assert_array_equal(converted_target, output_target.astype(int))


@pytest.mark.parametrize(
    "target, output_target, is_ova",
    [(np.array([0, 1, 1]), np.array([0, 1, 1]), False),
     (np.array([0, 1, 2]), np.array([0, 1, 2]), False),
     (np.array([[0, 1], [1, 0]]), np.array([1, 0]), True)]
)
def test_check_target_type_ova(target, output_target, is_ova):
    converted_target, binarize_target = check_target_type(
        target.astype(int), indicate_one_vs_all=True)
    assert_array_equal(converted_target, output_target.astype(int))
    assert binarize_target == is_ova


def test_check_target_warning():
    target = np.arange(4).reshape((2, 2))
    with pytest.warns(UserWarning, match='should be of types'):
        check_target_type(target)


def test_check_ratio_error():
    with raises(ValueError, match="'sampling_type' should be one of"):
        check_ratio('auto', np.array([1, 2, 3]), 'rnd')

    error_regex = "The target 'y' needs to have more than 1 class."
    with raises(ValueError, match=error_regex):
        check_ratio('auto', np.ones((10, )), 'over-sampling')

    error_regex = "When 'ratio' is a string, it needs to be one of"
    with raises(ValueError, match=error_regex):
        check_ratio('rnd', np.array([1, 2, 3]), 'over-sampling')


def test_ratio_all_over_sampling():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    for each in ('all', 'auto'):
        assert check_ratio(each, y, 'over-sampling') == {1: 50, 2: 0, 3: 75}


def test_ratio_all_under_sampling():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    ratio = check_ratio('all', y, 'under-sampling')
    assert ratio == {1: 25, 2: 25, 3: 25}


def test_ratio_majority_over_sampling():
    error_regex = "'ratio'='majority' cannot be used with over-sampler."
    with raises(ValueError, match=error_regex):
        check_ratio('majority', np.array([1, 2, 3]), 'over-sampling')


def test_ratio_majority_under_sampling():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    ratio = check_ratio('majority', y, 'under-sampling')
    assert ratio == {2: 25}


def test_ratio_not_minority_over_sampling():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    ratio = check_ratio('not minority', y, 'over-sampling')
    assert ratio == {1: 50, 2: 0}


def test_ratio_not_minority_under_sampling():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    ratio = check_ratio('not minority', y, 'under-sampling')
    assert ratio == {1: 25, 2: 25}
    ratio = check_ratio('auto', y, 'under-sampling')
    assert ratio == {1: 25, 2: 25}


def test_ratio_minority_over_sampling():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    ratio = check_ratio('minority', y, 'over-sampling')
    assert ratio == {3: 75}


def test_ratio_minority_under_sampling():
    error_regex = "'ratio'='minority' cannot be used with under-sampler."
    with raises(ValueError, match=error_regex):
        check_ratio('minority', np.array([1, 2, 3]), 'under-sampling')


def test_ratio_dict_error():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    ratio = {1: -100, 2: 50, 3: 25}
    with raises(ValueError, match="in a class cannot be negative."):
        check_ratio(ratio, y, 'under-sampling')
    ratio = {10: 10}
    with raises(ValueError, match="are not present in the data."):
        check_ratio(ratio, y, 'over-sampling')
    ratio = {1: 45, 2: 100, 3: 70}
    error_regex = ("With over-sampling methods, the number of samples in a"
                   " class should be greater or equal to the original number"
                   " of samples. Originally, there is 50 samples and 45"
                   " samples are asked.")
    with raises(ValueError, match=error_regex):
        check_ratio(ratio, y, 'over-sampling')

    error_regex = ("With under-sampling methods, the number of samples in a"
                   " class should be less or equal to the original number of"
                   " samples. Originally, there is 25 samples and 70 samples"
                   " are asked.")
    with raises(ValueError, match=error_regex):
                        check_ratio(ratio, y, 'under-sampling')


def test_ratio_dict_over_sampling():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    ratio = {1: 70, 2: 100, 3: 70}
    ratio_ = check_ratio(ratio, y, 'over-sampling')
    assert ratio_ == {1: 20, 2: 0, 3: 45}
    ratio = {1: 70, 2: 140, 3: 70}
    expected_msg = ("After over-sampling, the number of samples \(140\) in"
                    " class 2 will be larger than the number of samples in the"
                    " majority class \(class #2 -> 100\)")
    with warns(UserWarning, expected_msg):
        check_ratio(ratio, y, 'over-sampling')


def test_ratio_dict_under_sampling():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    ratio = {1: 30, 2: 45, 3: 25}
    ratio_ = check_ratio(ratio, y, 'under-sampling')
    assert ratio_ == ratio


def test_ratio_callable():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)

    def ratio_func(y):
        # this function could create an equal number of samples
        target_stats = Counter(y)
        n_samples = max(target_stats.values())
        return {key: int(n_samples)
                for key in target_stats.keys()}

    ratio_ = check_ratio(ratio_func, y, 'over-sampling')
    assert ratio_ == {1: 50, 2: 0, 3: 75}


def test_ratio_callable_args():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    multiplier = {1: 1.5, 2: 1, 3: 3}

    def ratio_func(y, multiplier):
        """samples such that each class will be affected by the multiplier."""
        target_stats = Counter(y)
        return {key: int(values * multiplier[key])
                for key, values in target_stats.items()}

    ratio_ = check_ratio(ratio_func, y, 'over-sampling',
                         multiplier=multiplier)
    assert ratio_ == {1: 25, 2: 0, 3: 50}


def test_hash_X_y():
    rng = check_random_state(0)
    X = rng.randn(2000, 20)
    y = np.array([0] * 500 + [1] * 1500)
    assert hash_X_y(X, y, 10, 10) == (joblib.hash(X[::200, ::2]),
                                      joblib.hash(y[::200]))

    X = rng.randn(5, 2)
    y = np.array([0] * 2 + [1] * 3)
    # all data will be used in this case
    assert hash_X_y(X, y) == (joblib.hash(X), joblib.hash(y))
