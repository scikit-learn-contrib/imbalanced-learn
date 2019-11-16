"""Common tests"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import pytest

from imblearn.utils.estimator_checks import check_estimator
from imblearn.utils.estimator_checks import _yield_all_checks
from imblearn.utils.testing import all_estimators


@pytest.mark.parametrize("name, Estimator", all_estimators())
def test_all_estimator_no_base_class(name, Estimator):
    # test that all_estimators doesn't find abstract classes.
    msg = (
        "Base estimators such as {0} should not be included"
        " in all_estimators"
    ).format(name)
    assert not name.lower().startswith("base"), msg


@pytest.mark.filterwarnings("ignore:'y' should be of types")
@pytest.mark.filterwarnings("ignore:The number of the samples to")
@pytest.mark.parametrize(
    "name, Estimator", all_estimators(include_meta_estimators=True)
)
def test_all_estimators(name, Estimator):
    # don't run twice the sampler tests. Meta-estimator do not have a
    # fit_resample method.
    check_estimator(Estimator, run_sampler_tests=False)


def _tested_non_meta_estimators():
    for name, Estimator in all_estimators():
        if name.startswith("_"):
            continue
        yield name, Estimator


def _generate_checks_per_estimator(check_generator, estimators):
    for name, Estimator in estimators:
        estimator = Estimator()
        for check in check_generator(name, estimator):
            yield name, Estimator, check


@pytest.mark.filterwarnings("ignore:'y' should be of types")
@pytest.mark.parametrize(
    "name, Estimator, check",
    _generate_checks_per_estimator(
        _yield_all_checks, _tested_non_meta_estimators()
    ),
)
def test_samplers(name, Estimator, check):
    # input validation etc for non-meta estimators
    check(name, Estimator)
