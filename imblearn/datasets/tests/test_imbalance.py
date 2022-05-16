"""Test the module easy ensemble."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from collections import Counter

import pytest
import numpy as np

from sklearn.datasets import load_iris

from imblearn.datasets import make_imbalance


@pytest.fixture
def iris():
    return load_iris(return_X_y=True)


@pytest.mark.parametrize(
    "sampling_strategy, err_msg",
    [
        ({0: -100, 1: 50, 2: 50}, "in a class cannot be negative"),
        ({0: 10, 1: 70}, "should be less or equal to the original"),
        ("random-string", "has to be a dictionary or a function"),
    ],
)
def test_make_imbalance_error(iris, sampling_strategy, err_msg):
    # we are reusing part of utils.check_sampling_strategy, however this is not
    # cover in the common tests so we will repeat it here
    X, y = iris
    with pytest.raises(ValueError, match=err_msg):
        make_imbalance(X, y, sampling_strategy=sampling_strategy)


def test_make_imbalance_error_single_class(iris):
    X, y = iris
    y = np.zeros_like(y)
    with pytest.raises(ValueError, match="needs to have more than 1 class."):
        make_imbalance(X, y, sampling_strategy={0: 10})


@pytest.mark.parametrize(
    "sampling_strategy, expected_counts",
    [
        ({0: 10, 1: 20, 2: 30}, {0: 10, 1: 20, 2: 30}),
        ({0: 10, 1: 20}, {0: 10, 1: 20, 2: 50}),
    ],
)
def test_make_imbalance_dict(iris, sampling_strategy, expected_counts):
    X, y = iris
    _, y_ = make_imbalance(X, y, sampling_strategy=sampling_strategy)
    assert Counter(y_) == expected_counts


@pytest.mark.parametrize("as_frame", [True, False], ids=["dataframe", "array"])
@pytest.mark.parametrize(
    "sampling_strategy, expected_counts",
    [
        (
            {"setosa": 10, "versicolor": 20, "virginica": 30},
            {"setosa": 10, "versicolor": 20, "virginica": 30},
        ),
        (
            {"setosa": 10, "versicolor": 20},
            {"setosa": 10, "versicolor": 20, "virginica": 50},
        ),
    ],
)
def test_make_imbalanced_iris(as_frame, sampling_strategy, expected_counts):
    pytest.importorskip("pandas")
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target
    y = iris.target_names[iris.target]
    X_res, y_res = make_imbalance(X, y, sampling_strategy=sampling_strategy)
    if as_frame:
        assert hasattr(X_res, "loc")
    assert Counter(y_res) == expected_counts
