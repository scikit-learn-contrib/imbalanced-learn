"""Test the module easy ensemble."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import print_function

from collections import Counter

import numpy as np

from pytest import raises

from sklearn.datasets import load_iris

from imblearn.datasets import make_imbalance

data = load_iris()
X, Y = data.data, data.target


def test_make_imbalanced_backcompat():
    # check an error is raised with we don't pass sampling_strategy and ratio
    with raises(TypeError, match="missing 1 required positional argument"):
        make_imbalance(X, Y)


def test_make_imbalance_error():
    # we are reusing part of utils.check_sampling_strategy, however this is not
    # cover in the common tests so we will repeat it here
    sampling_strategy = {0: -100, 1: 50, 2: 50}
    with raises(ValueError, match="in a class cannot be negative"):
        make_imbalance(X, Y, sampling_strategy)
    sampling_strategy = {0: 10, 1: 70}
    with raises(ValueError, match="should be less or equal to the original"):
        make_imbalance(X, Y, sampling_strategy)
    y_ = np.zeros((X.shape[0], ))
    sampling_strategy = {0: 10}
    with raises(ValueError, match="needs to have more than 1 class."):
        make_imbalance(X, y_, sampling_strategy)
    sampling_strategy = 'random-string'
    with raises(ValueError, match="has to be a dictionary or a function"):
        make_imbalance(X, Y, sampling_strategy)


def test_make_imbalance_dict():
    sampling_strategy = {0: 10, 1: 20, 2: 30}
    X_, y_ = make_imbalance(X, Y, sampling_strategy=sampling_strategy)
    assert Counter(y_) == sampling_strategy

    sampling_strategy = {0: 10, 1: 20}
    X_, y_ = make_imbalance(X, Y, sampling_strategy=sampling_strategy)
    assert Counter(y_) == {0: 10, 1: 20, 2: 50}


def test_make_imbalance_ratio():
    # check that using 'ratio' is working
    sampling_strategy = {0: 10, 1: 20, 2: 30}
    X_, y_ = make_imbalance(X, Y, ratio=sampling_strategy)
    assert Counter(y_) == sampling_strategy

    sampling_strategy = {0: 10, 1: 20}
    X_, y_ = make_imbalance(X, Y, ratio=sampling_strategy)
    assert Counter(y_) == {0: 10, 1: 20, 2: 50}
