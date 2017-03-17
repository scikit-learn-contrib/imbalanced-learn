from __future__ import division

import sys
import warnings
import traceback
import numpy as np

from sklearn.utils.estimator_checks import _yield_all_checks \
    as sklearn_yield_all_checks, check_estimator \
    as sklearn_check_estimator, check_parameters_default_constructible
from sklearn.utils.testing import (assert_warns, assert_raises_regex,
                                   assert_equal,
                                   set_random_state, SkipTest)
from sklearn.exceptions import SkipTestWarning, NotFittedError

from imblearn.base import SamplerMixin
from imblearn.utils.testing import binary_estimators, multiclass_estimators


def _yield_sampler_checks(name, Estimator):
    # Get only the name of binary and multiclass samplers
    binary_samplers = tuple([c[0] for c in binary_estimators()])
    multiclass_samplers = tuple([c[0] for c in multiclass_estimators()])
    if name in binary_samplers:
        yield check_continuous_warning
        yield check_multiclass_warning
    if name in multiclass_samplers:
        yield check_continuous_warning
    if 'ratio' in Estimator().get_params().keys():
        yield check_samplers_ratio_error
    yield check_samplers_one_label
    yield check_samplers_no_fit_error
    yield check_samplers_X_consistancy_sample
    yield check_samplers_fit


def _yield_all_checks(name, Estimator):
    # make the checks from scikit-learn
    sklearn_yield_all_checks(name, Estimator)
    # trigger our checks if this is a SamplerMixin
    if issubclass(Estimator, SamplerMixin):
        for check in _yield_sampler_checks(name, Estimator):
            yield check


def check_estimator(Estimator):
    """Check if estimator adheres to scikit-learn conventions and imblearn.

    This estimator will run an extensive test-suite for input validation,
    shapes, etc.
    Additional tests samplers if the Estimator inherits from the corresponding
    mixin from imblearn.base

    Parameters
    ----------
    Estimator : class
        Class to check. Estimator is a class object (not an instance).
    """
    name = Estimator.__name__
    # test scikit-learn compatibility
    sklearn_check_estimator(Estimator)
    check_parameters_default_constructible(name, Estimator)
    for check in _yield_all_checks(name, Estimator):
        try:
            check(name, Estimator)
        except SkipTest as message:
            # the only SkipTest thrown currently results from not
            # being able to import pandas.
            warnings.warn(message, SkipTestWarning)


def check_continuous_warning(name, Estimator):
    X = np.random.random((20, 2))
    y = np.linspace(0, 1, 20)
    estimator = Estimator()
    set_random_state(estimator)
    assert_warns(UserWarning, estimator.fit, X, y)


def check_multiclass_warning(name, Estimator):
    X = np.random.random((20, 2))
    y = np.array([0] * 3 + [1] * 2 + [2] * 15)
    estimator = Estimator()
    set_random_state(estimator)
    assert_warns(UserWarning, estimator.fit, X, y)


def check_samplers_one_label(name, Sampler):
    error_string_fit = "Sampler can't balance when only one class is present."
    sampler = Sampler()
    X = np.random.random((20, 2))
    y = np.zeros(20)
    try:
        sampler.fit(X, y)
    except ValueError as e:
        if 'class' not in repr(e):
            print(error_string_fit, Sampler, e)
            traceback.print_exc(file=sys.stdout)
            raise e
        else:
            return
    except Exception as exc:
        print(error_string_fit, traceback, exc)
        traceback.print_exc(file=sys.stdout)
        raise exc


def check_samplers_no_fit_error(name, Sampler):
    sampler = Sampler()
    X = np.random.random((20, 2))
    y = np.array([1] * 5 + [0] * 15)
    assert_raises_regex(NotFittedError, "instance is not fitted yet.",
                        sampler.sample, X, y)


def check_samplers_ratio_error(name, Sampler):
    sampler = Sampler()
    X = np.random.random((20, 2))
    y = np.array([1] * 5 + [0] * 15)

    ratio = 1000
    sampler.set_params(**{'ratio': ratio})
    assert_raises_regex(ValueError, "Ratio cannot be greater than one.",
                        sampler.fit, X, y)
    ratio = -1.0
    sampler.set_params(**{'ratio': ratio})
    assert_raises_regex(ValueError, "Ratio cannot be negative.",
                        sampler.fit, X, y)
    ratio = 'rnd'
    sampler.set_params(**{'ratio': ratio})
    assert_raises_regex(ValueError, "Unknown string for the parameter ratio.",
                        sampler.fit, X, y)
    ratio = [.5, .5]
    sampler.set_params(**{'ratio': ratio})
    assert_raises_regex(ValueError, "Unknown parameter type for ratio.",
                        sampler.fit, X, y)
    ratio = 1 / 1000
    sampler.set_params(**{'ratio': ratio})
    assert_raises_regex(RuntimeError, "The ratio requested at initialisation",
                        sampler.fit, X, y)


def check_samplers_X_consistancy_sample(name, Sampler):
    sampler = Sampler()
    X = np.random.random((20, 2))
    y = np.array([1] * 15 + [0] * 5)
    sampler.fit(X, y)
    X_different = np.random.random((30, 2))
    y_different = y = np.array([1] * 15 + [0] * 15)
    assert_raises_regex(RuntimeError, "to be the one earlier fitted",
                        sampler.sample, X_different, y_different)


def check_samplers_fit(name, Sampler):
    sampler = Sampler()
    X = np.random.random((20, 2))
    y = np.array([1] * 15 + [0] * 5)
    sampler.fit(X, y)
    assert_equal(sampler.min_c_, 0)
    assert_equal(sampler.maj_c_, 1)
    assert_equal(sampler.stats_c_[0], 5)
    assert_equal(sampler.stats_c_[1], 15)
