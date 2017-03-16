from sklearn.utils.testing import assert_greater
from sklearn.utils.testing import _named_check
from sklearn.utils.estimator_checks import check_estimator

from imblearn.utils.testing import all_estimators


def test_all_estimators():
    estimators = all_estimators(include_meta_estimators=True)
    assert_greater(len(estimators), 0)
    for name, Estimator in estimators:
        # some can just not be sensibly default constructed
        yield (_named_check(check_estimator, name),
               Estimator)
