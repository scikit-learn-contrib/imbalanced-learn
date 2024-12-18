"""Common tests"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import warnings
from collections import OrderedDict

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.utils._sklearn_compat import (
    parametrize_with_checks as parametrize_with_checks_sklearn,
)
from imblearn.utils._test_common.instance_generator import (
    _get_check_estimator_ids,
    _get_expected_failed_checks,
    _tested_estimators,
)
from imblearn.utils.estimator_checks import (
    _set_checking_parameters,
    check_dataframe_column_names_consistency,
    check_param_validation,
    parametrize_with_checks,
)
from imblearn.utils.testing import all_estimators


@pytest.mark.parametrize("name, Estimator", all_estimators())
def test_all_estimator_no_base_class(name, Estimator):
    # test that all_estimators doesn't find abstract classes.
    msg = f"Base estimators such as {name} should not be included in all_estimators"
    assert not name.lower().startswith("base"), msg


@parametrize_with_checks_sklearn(
    list(_tested_estimators()), expected_failed_checks=_get_expected_failed_checks
)
def test_estimators_compatibility_sklearn(estimator, check, request):
    _set_checking_parameters(estimator)
    check(estimator)


@parametrize_with_checks(
    list(_tested_estimators()), expected_failed_checks=_get_expected_failed_checks
)
def test_estimators_imblearn(estimator, check, request):
    # Common tests for estimator instances
    with ignore_warnings(
        category=(
            FutureWarning,
            ConvergenceWarning,
            UserWarning,
            FutureWarning,
        )
    ):
        _set_checking_parameters(estimator)
        check(estimator)


@pytest.mark.parametrize(
    "estimator", _tested_estimators(), ids=_get_check_estimator_ids
)
def test_check_param_validation(estimator):
    name = estimator.__class__.__name__
    _set_checking_parameters(estimator)
    check_param_validation(name, estimator)


@pytest.mark.parametrize("Sampler", [RandomOverSampler, RandomUnderSampler])
def test_strategy_as_ordered_dict(Sampler):
    """Check that it is possible to pass an `OrderedDict` as strategy."""
    rng = np.random.RandomState(42)
    X, y = rng.randn(30, 2), np.array([0] * 10 + [1] * 20)
    sampler = Sampler(random_state=42)
    if isinstance(sampler, RandomOverSampler):
        strategy = OrderedDict({0: 20, 1: 20})
    else:
        strategy = OrderedDict({0: 10, 1: 10})
    sampler.set_params(sampling_strategy=strategy)
    X_res, y_res = sampler.fit_resample(X, y)
    assert X_res.shape[0] == sum(strategy.values())
    assert y_res.shape[0] == sum(strategy.values())


@pytest.mark.parametrize(
    "estimator", _tested_estimators(), ids=_get_check_estimator_ids
)
def test_pandas_column_name_consistency(estimator):
    _set_checking_parameters(estimator)
    with ignore_warnings(category=(FutureWarning)):
        with warnings.catch_warnings(record=True) as record:
            check_dataframe_column_names_consistency(
                estimator.__class__.__name__, estimator
            )
        for warning in record:
            assert "was fitted without feature names" not in str(warning.message)
