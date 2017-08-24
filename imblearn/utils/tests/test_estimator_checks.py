"""Estimator tests - adapted from scikit-learn"""
import sys

import scipy.sparse as sp
import numpy as np
from pytest import raises

from sklearn.externals.six.moves import cStringIO as StringIO

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array

from imblearn.utils.estimator_checks import check_estimator


class CorrectNotFittedError(ValueError):
    """Exception class to raise if estimator is used before fitting.

    Like NotFittedError, it inherits from ValueError, but not from
    AttributeError. Used for testing only.
    """


class BaseBadClassifier(BaseEstimator, ClassifierMixin):
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.ones(X.shape[0])


class ChangesDict(BaseEstimator):
    def __init__(self):
        self.key = 0

    def fit(self, X, y=None):
        X, y = check_X_y(X, y)
        return self

    def predict(self, X):
        X = check_array(X)
        self.key = 1000
        return np.ones(X.shape[0])


class SetsWrongAttribute(BaseEstimator):
    def __init__(self):
        self.acceptable_key = 0

    def fit(self, X, y=None):
        self.wrong_attribute = 0
        X, y = check_X_y(X, y)
        return self


class ChangesWrongAttribute(BaseEstimator):
    def __init__(self):
        self.wrong_attribute = 0

    def fit(self, X, y=None):
        self.wrong_attribute = 1
        X, y = check_X_y(X, y)
        return self


class NoCheckinPredict(BaseBadClassifier):
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        return self


class NoSparseClassifier(BaseBadClassifier):
    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])
        if sp.issparse(X):
            raise ValueError("Nonsensical Error")
        return self

    def predict(self, X):
        X = check_array(X)
        return np.ones(X.shape[0])


class CorrectNotFittedErrorClassifier(BaseBadClassifier):
    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.coef_ = np.ones(X.shape[1])
        return self

    def predict(self, X):
        if not hasattr(self, 'coef_'):
            raise CorrectNotFittedError("estimator is not fitted yet")
        X = check_array(X)
        return np.ones(X.shape[0])


def test_check_estimator():
    # tests that the estimator actually fails on "bad" estimators.
    # not a complete test of all checks, which are very extensive.

    # check that we have a set_params and can clone
    msg = "it does not implement a 'get_params' methods"
    with raises(TypeError, match=msg):
        check_estimator(object)

    # check that we have a fit method
    msg = "object has no attribute 'fit'"
    with raises(AttributeError, match=msg):
        check_estimator(BaseEstimator)
    # check that fit does input validation
    msg = "TypeError not raised"
    with raises(AssertionError, match=msg):
        check_estimator(BaseBadClassifier)
    # check that predict does input validation (doesn't accept dicts in input)
    msg = "Estimator doesn't check for NaN and inf in predict"
    with raises(AssertionError, match=msg):
        check_estimator(NoCheckinPredict)
    # check that estimator state does not change
    # at transform/predict/predict_proba time
    msg = 'Estimator changes __dict__ during predict'
    with raises(AssertionError, match=msg):
        check_estimator(ChangesDict)
    # check that `fit` only changes attributes that
    # are private (start with an _ or end with a _).
    msg = ('Estimator changes public attribute\(s\) during the fit method.'
           ' Estimators are only allowed to change attributes started'
           ' or ended with _, but wrong_attribute changed')
    with raises(AssertionError, match=msg):
        check_estimator(ChangesWrongAttribute)
    # check that `fit` doesn't add any public attribute
    msg = ('Estimator adds public attribute\(s\) during the fit method.'
           ' Estimators are only allowed to add private attributes'
           ' either started with _ or ended'
           ' with _ but wrong_attribute added')
    with raises(AssertionError, match=msg):
        check_estimator(SetsWrongAttribute)
    # check for sparse matrix input handling
    name = NoSparseClassifier.__name__
    msg = ("Estimator " + name + " doesn't seem to fail gracefully on"
           " sparse data")
    # the check for sparse input handling prints to the stdout,
    # instead of raising an error, so as not to remove the original traceback.
    # that means we need to jump through some hoops to catch it.
    old_stdout = sys.stdout
    string_buffer = StringIO()
    sys.stdout = string_buffer
    try:
        check_estimator(NoSparseClassifier)
    except:
        pass
    finally:
        sys.stdout = old_stdout
    assert msg in string_buffer.getvalue()
