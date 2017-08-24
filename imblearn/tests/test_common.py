"""Common tests"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from sklearn.utils.testing import _named_check

from imblearn.utils.estimator_checks import check_estimator, _yield_all_checks
from imblearn.utils.testing import all_estimators


def test_all_estimator_no_base_class():
    # test that all_estimators doesn't find abstract classes.
    for name, Estimator in all_estimators():
        msg = ("Base estimators such as {0} should not be included"
               " in all_estimators").format(name)
        assert not name.lower().startswith('base'), msg


def test_all_estimators():
    estimators = all_estimators(include_meta_estimators=True)
    assert len(estimators) > 0
    for name, Estimator in estimators:
        # some can just not be sensibly default constructed
        yield (_named_check(check_estimator, name),
               Estimator)


def test_non_meta_estimators():
    # input validation etc for non-meta estimators
    estimators = all_estimators()
    for name, Estimator in estimators:
        if name.startswith("_"):
            continue
        for check in _yield_all_checks(name, Estimator):
            yield _named_check(check, name), name, Estimator
