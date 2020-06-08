"""Common tests"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import pytest

from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.estimator_checks import parametrize_with_checks as \
    parametrize_with_checks_sklearn
from sklearn.utils.estimator_checks import _construct_instance
from sklearn.utils._testing import ignore_warnings
from sklearn.utils._testing import set_random_state
from sklearn.utils._testing import SkipTest

from imblearn.utils.estimator_checks import parametrize_with_checks
# from imblearn.utils.estimator_checks import check_estimator
from imblearn.utils.estimator_checks import _set_checking_parameters
from imblearn.utils.estimator_checks import _yield_all_checks
from imblearn.utils.testing import all_estimators
from imblearn.under_sampling import NearMiss


@pytest.mark.parametrize("name, Estimator", all_estimators())
def test_all_estimator_no_base_class(name, Estimator):
    # test that all_estimators doesn't find abstract classes.
    msg = (
        f"Base estimators such as {name} should not be included"
        f" in all_estimators"
    )
    assert not name.lower().startswith("base"), msg


# @pytest.mark.filterwarnings("ignore:'y' should be of types")
# @pytest.mark.filterwarnings("ignore:The number of the samples to")
# @pytest.mark.parametrize(
#     "name, Estimator", all_estimators()
# )
# def test_all_estimators(name, Estimator):
#     # don't run twice the sampler tests. Meta-estimator do not have a
#     # fit_resample method.
#     check_estimator(Estimator, run_sampler_tests=False)


# def _tested_non_meta_estimators():
#     for name, Estimator in all_estimators():
#         if name.startswith("_"):
#             continue
#         yield name, Estimator


# def _generate_checks_per_estimator(check_generator, estimators):
#     for name, Estimator in estimators:
#         estimator = Estimator()
#         for check in check_generator(name, estimator):
#             yield name, Estimator, check


# @pytest.mark.filterwarnings("ignore:'y' should be of types")
# @pytest.mark.parametrize(
#     "name, Estimator, check",
#     _generate_checks_per_estimator(
#         _yield_all_checks, _tested_non_meta_estimators()
#     ),
# )
# def test_samplers(name, Estimator, check):
#     # input validation etc for non-meta estimators
#     check(name, Estimator)


def _tested_estimators():
    for name, Estimator in all_estimators():
        try:
            estimator = _construct_instance(Estimator)
            set_random_state(estimator)
        except SkipTest:
            continue

        if isinstance(estimator, NearMiss):
            for version in (1, 2, 3):
                yield clone(estimator).set_params(version=version)
        else:
            yield estimator


@parametrize_with_checks_sklearn(list(_tested_estimators()))
def test_estimators_sklearn(estimator, check, request):
    # Common tests for estimator instances
    with ignore_warnings(category=(FutureWarning,
                                   ConvergenceWarning,
                                   UserWarning, FutureWarning)):
        _set_checking_parameters(estimator)
        check(estimator)


@parametrize_with_checks(list(_tested_estimators()))
def test_estimators_imblearn(estimator, check, request):
    # Common tests for estimator instances
    with ignore_warnings(category=(FutureWarning,
                                   ConvergenceWarning,
                                   UserWarning, FutureWarning)):
        _set_checking_parameters(estimator)
        check(estimator)
