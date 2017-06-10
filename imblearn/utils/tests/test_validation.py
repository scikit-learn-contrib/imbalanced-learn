"""Test for the validation helper"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from collections import Counter

import numpy as np

from sklearn.neighbors.base import KNeighborsMixin
from sklearn.neighbors import NearestNeighbors

from sklearn.utils.testing import (assert_equal, assert_raises_regex,
                                   assert_warns_message)

from imblearn.utils import check_neighbors_object
from imblearn.utils import check_ratio


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


def test_check_ratio_error():
    assert_raises_regex(ValueError, "'sampling_type' should be one of",
                        check_ratio, 'auto', np.array([1, 2, 3]),
                        'rnd')
    assert_raises_regex(ValueError, "The target 'y' needs to have more than 1"
                        " class.", check_ratio, 'auto', np.ones((10, )),
                        'over-sampling')
    assert_raises_regex(ValueError, "When 'ratio' is a string, it needs to be"
                        " one of", check_ratio, 'rnd', np.array([1, 2, 3]),
                        'over-sampling')


def test_ratio_all_over_sampling():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    ratio = check_ratio('all', y, 'over-sampling')
    assert_equal(ratio, {1: 50, 2: 0, 3: 75})
    ratio = check_ratio('auto', y, 'over-sampling')
    assert_equal(ratio, {1: 50, 2: 0, 3: 75})


def test_ratio_all_under_sampling():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    ratio = check_ratio('all', y, 'under-sampling')
    assert_equal(ratio, {1: 25, 2: 25, 3: 25})


def test_ratio_majority_over_sampling():
    assert_raises_regex(ValueError, "'ratio'='majority' cannot be used with"
                        " over-sampler.", check_ratio, 'majority',
                        np.array([1, 2, 3]), 'over-sampling')


def test_ratio_majority_under_sampling():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    ratio = check_ratio('majority', y, 'under-sampling')
    assert_equal(ratio, {2: 25})


def test_ratio_not_minority_over_sampling():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    ratio = check_ratio('not minority', y, 'over-sampling')
    assert_equal(ratio, {1: 50, 2: 0})


def test_ratio_not_minority_under_sampling():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    ratio = check_ratio('not minority', y, 'under-sampling')
    assert_equal(ratio, {1: 25, 2: 25})
    ratio = check_ratio('auto', y, 'under-sampling')
    assert_equal(ratio, {1: 25, 2: 25})


def test_ratio_minority_over_sampling():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    ratio = check_ratio('minority', y, 'over-sampling')
    assert_equal(ratio, {3: 75})


def test_ratio_minority_under_sampling():
    assert_raises_regex(ValueError, "'ratio'='minority' cannot be used with"
                        " under-sampler.", check_ratio, 'minority',
                        np.array([1, 2, 3]), 'under-sampling')


def test_ratio_dict_error():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    ratio = {10: 10}
    assert_raises_regex(ValueError, "are not present in the data.",
                        check_ratio, ratio, y, 'over-sampling')
    ratio = {1: 45, 2: 100, 3: 70}
    assert_raises_regex(ValueError, "With over-sampling methods, the number"
                        " of samples in a class should be greater or equal"
                        " to the original number of samples. Originally,"
                        " there is 50 samples and 45 samples are asked.",
                        check_ratio, ratio, y, 'over-sampling')
    assert_raises_regex(ValueError, "With under-sampling methods, the number"
                        " of samples in a class should be less or equal"
                        " to the original number of samples. Originally,"
                        " there is 25 samples and 70 samples are asked.",
                        check_ratio, ratio, y, 'under-sampling')


def test_ratio_dict_over_sampling():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    ratio = {1: 70, 2: 100, 3: 70}
    ratio_ = check_ratio(ratio, y, 'over-sampling')
    assert_equal(ratio_, {1: 20, 2: 0, 3: 45})
    ratio = {1: 70, 2: 140, 3: 70}
    assert_warns_message(UserWarning, "After over-sampling, the number of"
                         " samples (140) in class 2 will be larger than the"
                         " number of samples in the majority class (class #2"
                         " -> 100)", check_ratio, ratio, y, 'over-sampling')


def test_ratio_dict_under_sampling():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    ratio = {1: 30, 2: 45, 3: 25}
    ratio_ = check_ratio(ratio, y, 'under-sampling')
    assert_equal(ratio_, ratio)


def test_ratio_float_error():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    ratio = -10
    assert_raises_regex(ValueError, "When 'ratio' is a float, it should in the"
                        " range", check_ratio, ratio, y, 'under-sampling')
    ratio = 10
    assert_raises_regex(ValueError, "When 'ratio' is a float, it should in the"
                        " range", check_ratio, ratio, y, 'under-sampling')


def test_ratio_float_over_sampling():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    ratio = 0.5
    ratio_ = check_ratio(ratio, y, 'over-sampling')
    assert_equal(ratio_, {1: 0, 3: 25})


def test_ratio_float_under_sampling():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    ratio = 0.5
    ratio_ = check_ratio(ratio, y, 'under-sampling')
    assert_equal(ratio_, {1: 50, 2: 50})


def test_ratio_callable():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)

    def ratio_func(y):
        # this function could create an equal number of samples
        target_stats = Counter(y)
        n_samples = max(target_stats.values())
        return {key: int(n_samples)
                for key in target_stats.keys()}

    ratio_ = check_ratio(ratio_func, y, 'over-sampling')
    assert_equal(ratio_, {1: 50, 2: 0, 3: 75})
