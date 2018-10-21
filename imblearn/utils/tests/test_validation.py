"""Test for the validation helper"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from collections import Counter
from collections import OrderedDict

import pytest
import numpy as np

from sklearn.neighbors.base import KNeighborsMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.testing import assert_array_equal

from imblearn.utils.testing import warns
from imblearn.utils import check_neighbors_object
from imblearn.utils import check_ratio
from imblearn.utils import check_sampling_strategy
from imblearn.utils import check_target_type

multiclass_target = np.array([1] * 50 + [2] * 100 + [3] * 25)
binary_target = np.array([1] * 25 + [0] * 100)


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
    estimator_cloned = check_neighbors_object(name, estimator)
    assert estimator.n_neighbors == estimator_cloned.n_neighbors
    n_neighbors = 'rnd'
    with pytest.raises(ValueError, match="has to be one of"):
        check_neighbors_object(name, n_neighbors)


@pytest.mark.parametrize("target, output_target", [(np.array(
    [0, 1, 1]), np.array([0, 1, 1])), (np.array([0, 1, 2]), np.array(
        [0, 1, 2])), (np.array([[0, 1], [1, 0]]), np.array([1, 0]))])
def test_check_target_type(target, output_target):
    converted_target = check_target_type(target.astype(int))
    assert_array_equal(converted_target, output_target.astype(int))


@pytest.mark.parametrize("target, output_target, is_ova",
                         [(np.array([0, 1, 1]), np.array([0, 1, 1]), False),
                          (np.array([0, 1, 2]), np.array([0, 1, 2]),
                           False), (np.array([[0, 1], [1, 0]]),
                                    np.array([1, 0]), True)])
def test_check_target_type_ova(target, output_target, is_ova):
    converted_target, binarize_target = check_target_type(
        target.astype(int), indicate_one_vs_all=True)
    assert_array_equal(converted_target, output_target.astype(int))
    assert binarize_target == is_ova


def test_check_sampling_strategy_warning():
    msg = 'dict for cleaning methods is deprecated'
    with pytest.warns(DeprecationWarning, match=msg):
        check_sampling_strategy({
            1: 0,
            2: 0,
            3: 0
        }, multiclass_target, 'clean-sampling')


def test_check_sampling_strategy_float_error():
    msg = "'clean-sampling' methods do let the user specify the sampling ratio"
    with pytest.raises(ValueError, match=msg):
        check_sampling_strategy(0.5, binary_target, 'clean-sampling')


def test_check_sampling_strategy_error():
    with pytest.raises(ValueError, match="'sampling_type' should be one of"):
        check_sampling_strategy('auto', np.array([1, 2, 3]), 'rnd')

    error_regex = "The target 'y' needs to have more than 1 class."
    with pytest.raises(ValueError, match=error_regex):
        check_sampling_strategy('auto', np.ones((10, )), 'over-sampling')

    error_regex = "When 'sampling_strategy' is a string, it needs to be one of"
    with pytest.raises(ValueError, match=error_regex):
        check_sampling_strategy('rnd', np.array([1, 2, 3]), 'over-sampling')


@pytest.mark.parametrize("sampling_strategy, sampling_type, err_msg",
                         [('majority', 'over-sampling', 'over-sampler'),
                          ('minority', 'under-sampling', 'under-sampler')])
def test_check_sampling_strategy_error_wrong_string(sampling_strategy,
                                                    sampling_type, err_msg):
    with pytest.raises(
            ValueError,
            match=("'{}' cannot be used with {}".format(
                sampling_strategy, err_msg))):
        check_sampling_strategy(sampling_strategy,
                                np.array([1, 2, 3]), sampling_type)


@pytest.mark.parametrize("sampling_strategy, sampling_method", [({
    10: 10
}, 'under-sampling'), ({
    10: 10
}, 'over-sampling'), ([10], 'clean-sampling')])
def test_sampling_strategy_class_target_unknown(sampling_strategy,
                                                sampling_method):
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    with pytest.raises(ValueError, match="are not present in the data."):
        check_sampling_strategy(sampling_strategy, y, sampling_method)


def test_sampling_strategy_dict_error():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    sampling_strategy = {1: -100, 2: 50, 3: 25}
    with pytest.raises(ValueError, match="in a class cannot be negative."):
        check_sampling_strategy(sampling_strategy, y, 'under-sampling')
    sampling_strategy = {1: 45, 2: 100, 3: 70}
    error_regex = ("With over-sampling methods, the number of samples in a"
                   " class should be greater or equal to the original number"
                   " of samples. Originally, there is 50 samples and 45"
                   " samples are asked.")
    with pytest.raises(ValueError, match=error_regex):
        check_sampling_strategy(sampling_strategy, y, 'over-sampling')

    error_regex = ("With under-sampling methods, the number of samples in a"
                   " class should be less or equal to the original number of"
                   " samples. Originally, there is 25 samples and 70 samples"
                   " are asked.")
    with pytest.raises(ValueError, match=error_regex):
        check_sampling_strategy(sampling_strategy, y, 'under-sampling')


@pytest.mark.parametrize("sampling_strategy", [-10, 10])
def test_sampling_strategy_float_error_not_in_range(sampling_strategy):
    y = np.array([1] * 50 + [2] * 100)
    with pytest.raises(ValueError, match='it should be in the range'):
        check_sampling_strategy(sampling_strategy, y, 'under-sampling')


def test_sampling_strategy_float_error_not_binary():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    with pytest.raises(ValueError, match='the type of target is binary'):
        sampling_strategy = 0.5
        check_sampling_strategy(sampling_strategy, y, 'under-sampling')


@pytest.mark.parametrize("sampling_method",
                         ['over-sampling', 'under-sampling'])
def test_sampling_strategy_list_error_not_clean_sampling(sampling_method):
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    with pytest.raises(ValueError, match='cannot be a list for samplers'):
        sampling_strategy = [1, 2, 3]
        check_sampling_strategy(sampling_strategy, y, sampling_method)


def _sampling_strategy_func(y):
    # this function could create an equal number of samples
    target_stats = Counter(y)
    n_samples = max(target_stats.values())
    return {key: int(n_samples) for key in target_stats.keys()}


@pytest.mark.parametrize(
    "sampling_strategy, sampling_type, expected_sampling_strategy, target",
    [('auto', 'under-sampling', {
        1: 25,
        2: 25
    }, multiclass_target), ('auto', 'clean-sampling', {
        1: 25,
        2: 25
    }, multiclass_target), ('auto', 'over-sampling', {
        1: 50,
        3: 75
    }, multiclass_target), ('all', 'over-sampling', {
        1: 50,
        2: 0,
        3: 75
    }, multiclass_target), ('all', 'under-sampling', {
        1: 25,
        2: 25,
        3: 25
    }, multiclass_target), ('all', 'clean-sampling', {
        1: 25,
        2: 25,
        3: 25
    }, multiclass_target), ('majority', 'under-sampling', {
        2: 25
    }, multiclass_target), ('majority', 'clean-sampling', {
        2: 25
    }, multiclass_target), ('minority', 'over-sampling', {
        3: 75
    }, multiclass_target), ('not minority', 'over-sampling', {
        1: 50,
        2: 0
    }, multiclass_target), ('not minority', 'under-sampling', {
        1: 25,
        2: 25
    }, multiclass_target), ('not minority', 'clean-sampling', {
        1: 25,
        2: 25
    }, multiclass_target), ('not majority', 'over-sampling', {
        1: 50,
        3: 75
    }, multiclass_target), ('not majority', 'under-sampling', {
        1: 25,
        3: 25
    }, multiclass_target), ('not majority', 'clean-sampling', {
        1: 25,
        3: 25
    }, multiclass_target), ({
        1: 70,
        2: 100,
        3: 70
    }, 'over-sampling', {
        1: 20,
        2: 0,
        3: 45
    }, multiclass_target), ({
        1: 30,
        2: 45,
        3: 25
    }, 'under-sampling', {
        1: 30,
        2: 45,
        3: 25
    }, multiclass_target), ([1], 'clean-sampling', {
        1: 25
    }, multiclass_target), (_sampling_strategy_func, 'over-sampling', {
        1: 50,
        2: 0,
        3: 75
    }, multiclass_target), (0.5, 'over-sampling', {
        1: 25
    }, binary_target), (0.5, 'under-sampling', {
        0: 50
    }, binary_target)])
def test_check_sampling_strategy(sampling_strategy, sampling_type,
                                 expected_sampling_strategy, target):
    sampling_strategy_ = check_sampling_strategy(sampling_strategy, target,
                                                 sampling_type)
    assert sampling_strategy_ == expected_sampling_strategy


@pytest.mark.parametrize("ratio, sampling_type, expected_ratio, target", [
    ('auto', 'under-sampling', {
        1: 25,
        2: 25
    }, multiclass_target), ('auto', 'clean-sampling', {
        1: 25,
        2: 25
    }, multiclass_target), ('auto', 'over-sampling', {
        1: 50,
        3: 75
    }, multiclass_target), ('all', 'over-sampling', {
        1: 50,
        2: 0,
        3: 75
    }, multiclass_target), ('all', 'under-sampling', {
        1: 25,
        2: 25,
        3: 25
    }, multiclass_target), ('all', 'clean-sampling', {
        1: 25,
        2: 25,
        3: 25
    }, multiclass_target), ('majority', 'under-sampling', {
        2: 25
    }, multiclass_target), ('majority', 'clean-sampling', {
        2: 25
    }, multiclass_target), ('minority', 'over-sampling', {
        3: 75
    }, multiclass_target), ('not minority', 'over-sampling', {
        1: 50,
        2: 0
    }, multiclass_target), ('not minority', 'under-sampling', {
        1: 25,
        2: 25
    }, multiclass_target), ('not minority', 'clean-sampling', {
        1: 25,
        2: 25
    }, multiclass_target), ('not majority', 'over-sampling', {
        1: 50,
        3: 75
    }, multiclass_target), ('not majority', 'under-sampling', {
        1: 25,
        3: 25
    }, multiclass_target), ('not majority', 'clean-sampling', {
        1: 25,
        3: 25
    }, multiclass_target), ({
        1: 70,
        2: 100,
        3: 70
    }, 'over-sampling', {
        1: 20,
        2: 0,
        3: 45
    }, multiclass_target), ({
        1: 30,
        2: 45,
        3: 25
    }, 'under-sampling', {
        1: 30,
        2: 45,
        3: 25
    }, multiclass_target), ([1], 'clean-sampling', {
        1: 25
    }, multiclass_target), (_sampling_strategy_func, 'over-sampling', {
        1: 50,
        2: 0,
        3: 75
    }, multiclass_target), (0.5, 'over-sampling', {
        1: 25
    }, binary_target), (0.5, 'under-sampling', {
        0: 50
    }, binary_target)
])
def test_check_ratio(ratio, sampling_type, expected_ratio, target):
    with pytest.warns(DeprecationWarning, match="check_ratio is deprecated"):
        ratio_ = check_ratio(ratio, target, sampling_type)
        assert ratio_ == expected_ratio


def test_sampling_strategy_dict_over_sampling():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    sampling_strategy = {1: 70, 2: 140, 3: 70}
    expected_msg = ("After over-sampling, the number of samples \(140\) in"
                    " class 2 will be larger than the number of samples in the"
                    " majority class \(class #2 -> 100\)")
    with warns(UserWarning, expected_msg):
        check_sampling_strategy(sampling_strategy, y, 'over-sampling')


def test_sampling_strategy_callable_args():
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    multiplier = {1: 1.5, 2: 1, 3: 3}

    def sampling_strategy_func(y, multiplier):
        """samples such that each class will be affected by the multiplier."""
        target_stats = Counter(y)
        return {
            key: int(values * multiplier[key])
            for key, values in target_stats.items()
        }

    sampling_strategy_ = check_sampling_strategy(
        sampling_strategy_func, y, 'over-sampling', multiplier=multiplier)
    assert sampling_strategy_ == {1: 25, 2: 0, 3: 50}


@pytest.mark.parametrize(
    "sampling_strategy, sampling_type, expected_result",
    [({3: 25, 1: 25, 2: 25}, 'under-sampling',
      OrderedDict({1: 25, 2: 25, 3: 25})),
     ({3: 100, 1: 100, 2: 100}, 'over-sampling',
      OrderedDict({1: 50, 2: 0, 3: 75}))])
def test_sampling_strategy_check_order(sampling_strategy, sampling_type,
                                       expected_result):
    # We pass on purpose a non sorted dictionary and check that the resulting
    # dictionary is sorted. Refer to issue #428.
    y = np.array([1] * 50 + [2] * 100 + [3] * 25)
    sampling_strategy_ = check_sampling_strategy(
        sampling_strategy, y, sampling_type)
    assert sampling_strategy_ == expected_result
