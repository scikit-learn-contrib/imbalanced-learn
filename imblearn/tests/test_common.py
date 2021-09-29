"""Common tests"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import pytest

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.estimator_checks import (
    parametrize_with_checks as parametrize_with_checks_sklearn,
)
from sklearn.utils.estimator_checks import _construct_instance
from sklearn.utils._testing import ignore_warnings
from sklearn.utils._testing import set_random_state
from sklearn.utils._testing import SkipTest

from imblearn.utils.estimator_checks import parametrize_with_checks
from imblearn.utils.estimator_checks import _set_checking_parameters
from imblearn.utils.testing import all_estimators
from imblearn.under_sampling import NearMiss


@pytest.mark.parametrize("name, Estimator", all_estimators())
def test_all_estimator_no_base_class(name, Estimator):
    # test that all_estimators doesn't find abstract classes.
    msg = f"Base estimators such as {name} should not be included" f" in all_estimators"
    assert not name.lower().startswith("base"), msg


def _tested_estimators():
    for name, Estimator in all_estimators():
        try:
            estimator = _construct_instance(Estimator)
            set_random_state(estimator)
        except SkipTest:
            continue

        if isinstance(estimator, NearMiss):
            # For NearMiss, let's check the three algorithms
            for version in (1, 2, 3):
                yield clone(estimator).set_params(version=version)
        else:
            yield estimator


@parametrize_with_checks_sklearn(list(_tested_estimators()))
def test_estimators_compatibility_sklearn(estimator, check, request):
    _set_checking_parameters(estimator)
    check(estimator)


@parametrize_with_checks(list(_tested_estimators()))
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
