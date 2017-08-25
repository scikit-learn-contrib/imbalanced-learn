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


def test_make_imbalance_error():
    # we are reusing part of utils.check_ratio, however this is not cover in
    # the common tests so we will repeat it here
    ratio = {0: -100, 1: 50, 2: 50}
    with raises(ValueError, match="in a class cannot be negative"):
        make_imbalance(X, Y, ratio)
    ratio = {0: 10, 1: 70}
    with raises(ValueError, match="should be less or equal to the original"):
        make_imbalance(X, Y, ratio)
    y_ = np.zeros((X.shape[0], ))
    ratio = {0: 10}
    with raises(ValueError, match="needs to have more than 1 class."):
        make_imbalance(X, y_, ratio)
    ratio = 'random-string'
    with raises(ValueError, match="has to be a dictionary or a function"):
        make_imbalance(X, Y, ratio)


def test_make_imbalance_dict():
    ratio = {0: 10, 1: 20, 2: 30}
    X_, y_ = make_imbalance(X, Y, ratio=ratio)
    assert Counter(y_) == ratio

    ratio = {0: 10, 1: 20}
    X_, y_ = make_imbalance(X, Y, ratio=ratio)
    assert Counter(y_) == {0: 10, 1: 20, 2: 50}
