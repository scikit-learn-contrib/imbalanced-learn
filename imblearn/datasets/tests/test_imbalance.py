"""Test the module easy ensemble."""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT


from __future__ import print_function

from collections import Counter

import numpy as np

from sklearn.datasets import load_iris
from sklearn.utils.testing import (assert_equal, assert_raises_regex,
                                   assert_warns_message)

from imblearn.datasets import make_imbalance

data = load_iris()
X, Y = data.data, data.target


def test_make_imbalance_error():
    # we are reusing part of utils.check_ratio, however this is not cover in
    # the common tests so we will repeat it here
    ratio = {0: -100, 1: 50, 2: 50}
    assert_raises_regex(ValueError, "in a class cannot be negative",
                        make_imbalance, X, Y, ratio)
    ratio = {0: 10, 1: 70}
    assert_raises_regex(ValueError, "should be less or equal to the original",
                        make_imbalance, X, Y, ratio)
    y_ = np.zeros((X.shape[0], ))
    ratio = {0: 10}
    assert_raises_regex(ValueError, "needs to have more than 1 class.",
                        make_imbalance, X, y_, ratio)
    ratio = 'random-string'
    assert_raises_regex(ValueError, "has to be a dictionary or a function",
                        make_imbalance, X, Y, ratio)


# FIXME: to be removed in 0.4 due to deprecation
def test_make_imbalance_float():
    X_, y_ = assert_warns_message(DeprecationWarning,
                                  "'min_c_' is deprecated in 0.2",
                                  make_imbalance, X, Y, ratio=0.5, min_c_=1)
    X_, y_ = assert_warns_message(DeprecationWarning,
                                  "'ratio' being a float is deprecated",
                                  make_imbalance, X, Y, ratio=0.5, min_c_=1)
    assert_equal(Counter(y_), {0: 50, 1: 25, 2: 50})
    # resample without using min_c_
    X_, y_ = make_imbalance(X_, y_, ratio=0.25, min_c_=None)
    assert_equal(Counter(y_), {0: 50, 1: 12, 2: 50})


def test_make_imbalance_dict():
    ratio = {0: 10, 1: 20, 2: 30}
    X_, y_ = make_imbalance(X, Y, ratio=ratio)
    assert_equal(Counter(y_), ratio)

    ratio = {0: 10, 1: 20}
    X_, y_ = make_imbalance(X, Y, ratio=ratio)
    assert_equal(Counter(y_), {0: 10, 1: 20, 2: 50})
