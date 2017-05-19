"""Utils to check the samplers and compatibility with scikit-learn"""

# Adapated from scikit-learn
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

from __future__ import division

import sys
import traceback

from collections import Counter

import numpy as np

from sklearn.datasets import make_classification
from sklearn.utils.estimator_checks import _yield_all_checks \
    as sklearn_yield_all_checks, check_estimator \
    as sklearn_check_estimator, check_parameters_default_constructible
from sklearn.exceptions import NotFittedError
from sklearn.utils.testing import (assert_warns, assert_raises_regex,
                                   assert_true, set_random_state,
                                   assert_equal)

from imblearn.base import SamplerMixin
from imblearn.over_sampling.base import BaseOverSampler
from imblearn.under_sampling.base import BaseCleaningSampler, BaseUnderSampler
from imblearn.ensemble.base import BaseEnsembleSampler


def _yield_sampler_checks(name, Estimator):
    yield check_continuous_warning
    yield check_samplers_one_label
    yield check_samplers_no_fit_error
    yield check_samplers_X_consistancy_sample
    yield check_samplers_fit
    yield check_samplers_fit_sample
    yield check_samplers_ratio_fit_sample


def _yield_all_checks(name, Estimator):
    # make the checks from scikit-learn
    sklearn_yield_all_checks(name, Estimator)
    # trigger our checks if this is a SamplerMixin
    if issubclass(Estimator, SamplerMixin):
        for check in _yield_sampler_checks(name, Estimator):
            yield check
    # FIXME already present in scikit-learn 0.19
    yield check_dont_overwrite_parameters


def check_estimator(Estimator):
    """Check if estimator adheres to scikit-learn conventions and
    imbalanced-learn

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
        check(name, Estimator)


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


def multioutput_estimator_convert_y_2d(name, y):
    # Estimators in mono_output_task_error raise ValueError if y is of 1-D
    # Convert into a 2-D y for those estimators.
    if "MultiTask" in name:
        return np.reshape(y, (-1, 1))
    return y


def is_public_parameter(attr):
    return not (attr.startswith('_') or attr.endswith('_'))


def check_dont_overwrite_parameters(name, Estimator):
    # check that fit method only changes or sets private attributes
    if hasattr(Estimator.__init__, "deprecated_original"):
        # to not check deprecated classes
        return
    rnd = np.random.RandomState(0)
    X = 3 * rnd.uniform(size=(20, 3))
    y = X[:, 0].astype(np.int)
    y = multioutput_estimator_convert_y_2d(name, y)
    estimator = Estimator()

    if hasattr(estimator, "n_components"):
        estimator.n_components = 1
    if hasattr(estimator, "n_clusters"):
        estimator.n_clusters = 1

    set_random_state(estimator, 1)
    dict_before_fit = estimator.__dict__.copy()
    estimator.fit(X, y)

    dict_after_fit = estimator.__dict__

    public_keys_after_fit = [key for key in dict_after_fit.keys()
                             if is_public_parameter(key)]

    attrs_added_by_fit = [key for key in public_keys_after_fit
                          if key not in dict_before_fit.keys()]

    # check that fit doesn't add any public attribute
    assert_true(not attrs_added_by_fit,
                ('Estimator adds public attribute(s) during'
                 ' the fit method.'
                 ' Estimators are only allowed to add private attributes'
                 ' either started with _ or ended'
                 ' with _ but %s added' % ', '.join(attrs_added_by_fit)))

    # check that fit doesn't change any public attribute
    attrs_changed_by_fit = [key for key in public_keys_after_fit
                            if (dict_before_fit[key]
                                is not dict_after_fit[key])]

    assert_true(not attrs_changed_by_fit,
                ('Estimator changes public attribute(s) during'
                 ' the fit method. Estimators are only allowed'
                 ' to change attributes started'
                 ' or ended with _, but'
                 ' %s changed' % ', '.join(attrs_changed_by_fit)))


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


def check_samplers_X_consistancy_sample(name, Sampler):
    sampler = Sampler()
    X = np.random.random((30, 2))
    y = np.array([1] * 20 + [0] * 10)
    sampler.fit(X, y)
    X_different = np.random.random((40, 2))
    y_different = y = np.array([1] * 25 + [0] * 15)
    assert_raises_regex(RuntimeError, "X and y need to be same array earlier",
                        sampler.sample, X_different, y_different)


def check_samplers_fit(name, Sampler):
    sampler = Sampler()
    X = np.random.random((30, 2))
    y = np.array([1] * 20 + [0] * 10)
    sampler.fit(X, y)
    assert_true(hasattr(sampler, 'ratio_'))


def check_samplers_fit_sample(name, Sampler):
    sampler = Sampler(random_state=0)
    X, y = make_classification(n_samples=1000, n_classes=3,
                               n_informative=4, weights=[0.2, 0.3, 0.5],
                               random_state=0)
    target_stats = Counter(y)
    X_res, y_res = sampler.fit_sample(X, y)
    if isinstance(sampler, BaseOverSampler):
        target_stats_res = Counter(y_res)
        n_samples = max(target_stats.values())
        assert_true(all(value >= n_samples
                        for value in Counter(y_res).values()))
    elif isinstance(sampler, BaseUnderSampler):
        n_samples = min(target_stats.values())
        assert_true(all(value == n_samples
                        for value in Counter(y_res).values()))
    elif isinstance(sampler, BaseCleaningSampler):
        target_stats_res = Counter(y_res)
        class_minority = min(target_stats, key=target_stats.get)
        assert_true(
            all(target_stats[class_sample] > target_stats_res[class_sample]
                for class_sample in target_stats.keys()
                if class_sample != class_minority))
    elif isinstance(sampler, BaseEnsembleSampler):
        y_ensemble = y_res[0]
        n_samples = min(target_stats.values())
        assert_true(all(value == n_samples
                        for value in Counter(y_ensemble).values()))


def check_samplers_ratio_fit_sample(name, Sampler):
    # in this test we will force all samplers to not change the class 1
    X, y = make_classification(n_samples=1000, n_classes=3,
                               n_informative=4, weights=[0.2, 0.3, 0.5],
                               random_state=0)
    target_stats = Counter(y)
    sampler = Sampler(random_state=0)
    if isinstance(sampler, BaseOverSampler):
        ratio = {2: 498, 0: 498}
        sampler.set_params(ratio=ratio)
        X_res, y_res = sampler.fit_sample(X, y)
        assert_equal(target_stats[1], Counter(y_res)[1])
    elif isinstance(sampler, BaseUnderSampler):
        ratio = {2: 201, 0: 201}
        sampler.set_params(ratio=ratio)
        X_res, y_res = sampler.fit_sample(X, y)
        assert_equal(target_stats[1], Counter(y_res)[1])
    elif isinstance(sampler, BaseCleaningSampler):
        ratio = {2: 201, 0: 201}
        sampler.set_params(ratio=ratio)
        X_res, y_res = sampler.fit_sample(X, y)
        assert_equal(target_stats[1], Counter(y_res)[1])
    elif isinstance(sampler, BaseEnsembleSampler):
        ratio = {2: 201, 0: 201}
        sampler.set_params(ratio=ratio)
        X_res, y_res = sampler.fit_sample(X, y)
        y_ensemble = y_res[0]
        assert_equal(target_stats[1], Counter(y_ensemble)[1])
