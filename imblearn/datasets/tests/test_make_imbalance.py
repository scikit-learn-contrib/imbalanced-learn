"""Test the module easy ensemble."""

from __future__ import print_function

from collections import Counter

import numpy as np
from sklearn.utils.testing import assert_true
from numpy.testing import assert_equal, assert_raises_regex

from imblearn.datasets import make_imbalance

# Generate a global dataset to use
X = np.random.random((1000, 2))
Y = np.zeros(1000)
Y[500:] = 1


def test_make_imbalance_bad_ratio():
    min_c_ = 1

    # Define a zero ratio
    ratio = 0.0
    assert_raises_regex(ValueError, "Ratio have to be strictly positive",
                        make_imbalance, X, Y, ratio, min_c_)

    # Define a negative ratio
    ratio = -2.0
    assert_raises_regex(ValueError, "Ratio have to be strictly positive",
                        make_imbalance, X, Y, ratio, min_c_)

    # Define a ratio greater than 1
    ratio = 2.0
    assert_raises_regex(ValueError, "Ratio cannot be greater than one",
                        make_imbalance, X, Y, ratio, min_c_)

    # Define ratio as a list which is not supported
    ratio = [.5, .5]
    assert_raises_regex(ValueError, "Ratio must be a float between",
                        make_imbalance, X, Y, ratio, min_c_)


def test_make_imbalance_invalid_ratio():
    y_ = np.zeros((X.shape[0], ))
    y_[0] = 1

    ratio = 0.5
    assert_raises_regex(ValueError, "Current imbalance ratio of data",
                        make_imbalance, X, y_, ratio)


def test_make_imbalance_single_class():
    y_ = np.zeros((X.shape[0], ))
    ratio = 0.5
    assert_raises_regex(ValueError, "Not enough samples for desired ratio!",
                        make_imbalance, X, y_, ratio)


def test_make_imbalance_1():
    X_, y_ = make_imbalance(X, Y, ratio=0.5, min_c_=1)
    counter = Counter(y_)
    assert_equal(counter[0], 500)
    assert_equal(counter[1], 250)
    assert_true(np.all([X_i in X for X_i in X_]))


def test_make_imbalance_2():
    X_, y_ = make_imbalance(X, Y, ratio=0.25, min_c_=1)
    counter = Counter(y_)
    assert_equal(counter[0], 500)
    assert_equal(counter[1], 125)
    assert_true(np.all([X_i in X for X_i in X_]))


def test_make_imbalance_3():
    X_, y_ = make_imbalance(X, Y, ratio=0.1, min_c_=1)
    counter = Counter(y_)
    assert_equal(counter[0], 500)
    assert_equal(counter[1], 50)
    assert_true(np.all([X_i in X for X_i in X_]))


def test_make_imbalance_4():
    X_, y_ = make_imbalance(X, Y, ratio=0.01, min_c_=1)
    counter = Counter(y_)
    assert_equal(counter[0], 500)
    assert_equal(counter[1], 5)
    assert_true(np.all([X_i in X for X_i in X_]))


def test_make_imbalance_5():
    X_, y_ = make_imbalance(X, Y, ratio=0.01, min_c_=0)
    counter = Counter(y_)
    assert_equal(counter[1], 500)
    assert_equal(counter[0], 5)
    assert_true(np.all([X_i in X for X_i in X_]))


def test_make_imbalance_multiclass():
    # Make y to be multiclass
    y_ = np.zeros(1000)
    y_[100:500] = 1
    y_[500:] = 2

    # Resample the data
    X_, y_ = make_imbalance(X, y_, ratio=0.1, min_c_=0)
    counter = Counter(y_)
    assert_equal(counter[0], 90)
    assert_equal(counter[1], 400)
    assert_equal(counter[2], 500)
    assert_true(np.all([X_i in X for X_i in X_]))
