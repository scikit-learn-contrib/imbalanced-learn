"""Testing the metric for classification with imbalanced dataset"""

from __future__ import division, print_function

from functools import partial

import numpy as np

from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_no_warnings, assert_equal,
                           assert_almost_equal, assert_raises)
from sklearn.utils.testing import assert_warns_message, ignore_warnings

from sklearn import datasets
from sklearn import svm

from sklearn.preprocessing import label_binarize
from sklearn.utils.testing import assert_not_equal
from sklearn.utils.validation import check_random_state

from imblearn.metrics import sensitivity_specificity_support
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score

RND_SEED = 42

###############################################################################
# Utilities for testing


def make_prediction(dataset=None, binary=False):
    """Make some classification predictions on a toy dataset using a SVC
    If binary is True restrict to a binary classification problem instead of a
    multiclass classification problem
    """

    if dataset is None:
        # import some data to play with
        dataset = datasets.load_iris()

    X = dataset.data
    y = dataset.target

    if binary:
        # restrict to a binary classification task
        X, y = X[y < 2], y[y < 2]

    n_samples, n_features = X.shape
    p = np.arange(n_samples)

    rng = check_random_state(37)
    rng.shuffle(p)
    X, y = X[p], y[p]
    half = int(n_samples / 2)

    # add noisy features to make the problem harder and avoid perfect results
    rng = np.random.RandomState(0)
    X = np.c_[X, rng.randn(n_samples, 200 * n_features)]

    # run classifier, get class probabilities and label predictions
    clf = svm.SVC(kernel='linear', probability=True, random_state=0)
    probas_pred = clf.fit(X[:half], y[:half]).predict_proba(X[half:])

    if binary:
        # only interested in probabilities of the positive case
        # XXX: do we really want a special API for the binary case?
        probas_pred = probas_pred[:, 1]

    y_pred = clf.predict(X[half:])
    y_true = y[half:]

    return y_true, y_pred, probas_pred


###############################################################################
# Tests

def test_sensitivity_specificity_support_binary():
    """Test the sensitivity specificity for binary classification task"""
    y_true, y_pred, _ = make_prediction(binary=True)

    # detailed measures for each class
    sens, spec, supp = sensitivity_specificity_support(y_true, y_pred,
                                                       average=None)
    assert_array_almost_equal(sens, [0.88, 0.68], 2)
    assert_array_almost_equal(spec, [0.68, 0.88], 2)
    assert_array_equal(supp, [25, 25])

    # individual scoring function that can be used for grid search: in the
    # binary class case the score is the value of the measure for the positive
    # class (e.g. label == 1). This is deprecated for average != 'binary'.
    for kwargs, my_assert in [({}, assert_no_warnings),
                              ({'average': 'binary'}, assert_no_warnings)]:
        sens = my_assert(sensitivity_score, y_true, y_pred, **kwargs)
        assert_array_almost_equal(sens, 0.68, 2)

        spec = my_assert(specificity_score, y_true, y_pred, **kwargs)
        assert_array_almost_equal(spec, 0.88, 2)


def test_sensitivity_specificity_binary_single_class():
    # Test sensitivity and specificity score behave with a single positive or
    # negative class
    # Such a case may occur with non-stratified cross-validation
    assert_equal(1., sensitivity_score([1, 1], [1, 1]))
    assert_equal(0., specificity_score([1, 1], [1, 1]))

    assert_equal(0., sensitivity_score([-1, -1], [-1, -1]))
    assert_equal(0., specificity_score([-1, -1], [-1, -1]))


def test_sensitivity_specificity_error_multilabels():
    # Test either if an error is raised when the input are multilabels
    y_true = [1, 3, 3, 2]
    y_pred = [1, 1, 3, 2]
    y_true_bin = label_binarize(y_true, classes=np.arange(5))
    y_pred_bin = label_binarize(y_pred, classes=np.arange(5))

    assert_raises(ValueError, sensitivity_score, y_true_bin, y_pred_bin)

@ignore_warnings
def test_sensitivity_specifiicity_extra_labels():
    # Test handling of explicit additional (not in input) labels to SS
    y_true = [1, 3, 3, 2]
    y_pred = [1, 1, 3, 2]

    actual = sensitivity_score(y_true, y_pred, labels=[0, 1, 2, 3, 4],
                               average=None)
    assert_array_almost_equal([0., 1., 1., .5, 0.], actual)

    # Macro average is changed
    actual = sensitivity_score(y_true, y_pred, labels=[0, 1, 2, 3, 4],
                               average='macro')
    assert_array_almost_equal(np.mean([0., 1., 1., .5, 0.]), actual)

    # Weighted average is changed
    assert_almost_equal(sensitivity_score(y_true, y_pred,
                                          labels=[0, 1, 2, 3, 4],
                                          average='weighted'),
                        sensitivity_score(y_true, y_pred, labels=None,
                                          average='weighted'))

@ignore_warnings
def test_sensitivity_specificity_f_ignored_labels():
    # Test a subset of labels may be requested for SS
    y_true = [1, 1, 2, 3]
    y_pred = [1, 3, 3, 3]

    sensitivity_13 = partial(sensitivity_score, y_true, y_pred, labels=[1, 3])
    sensitivity_all = partial(sensitivity_score, y_true, y_pred, labels=None)

    assert_array_almost_equal([.5, 1.], sensitivity_13(average=None))
    assert_almost_equal((.5 + 1.) / 2, sensitivity_13(average='macro'))
    assert_almost_equal((.5 * 2 + 1. * 1) / 3,
                        sensitivity_13(average='weighted'))

    # ensure the above were meaningful tests:
    for average in ['macro', 'weighted']:
        assert_not_equal(sensitivity_13(average=average),
                         sensitivity_all(average=average))


@ignore_warnings
def test_sensitivity_specificity_support_errors():
    y_true, y_pred, _ = make_prediction(binary=True)

    # Bad pos_label
    assert_raises(ValueError, sensitivity_specificity_support,
                  y_true, y_pred, pos_label=2, average='binary')

    # Bad average option
    assert_raises(ValueError, sensitivity_specificity_support,
                  [0, 1, 2], [1, 2, 0], average='mega')


def test_sensitivity_specificity_unused_pos_label():
    # Check warning that pos_label unused when set to non-default value
    # but average != 'binary'; even if data is binary.
    assert_warns_message(UserWarning,
                         "Note that pos_label (set to 2) is "
                         "ignored when average != 'binary' (got 'macro'). You "
                         "may use labels=[pos_label] to specify a single "
                         "positive class.", sensitivity_specificity_support,
                         [1, 2, 1], [1, 2, 2], pos_label=2, average='macro')


def test_sensitivity_specificity_multiclass():
    # Test Precision Recall and F1 Score for multiclass classification task
    y_true, y_pred, _ = make_prediction(binary=False)

    # compute scores with default labels introspection
    sens, spec, supp = sensitivity_specificity_support(y_true, y_pred,
                                                       average=None)
    assert_array_almost_equal(spec, [0.92, 0.86, 0.55], 2)
    assert_array_almost_equal(sens, [0.79, 0.09, 0.90], 2)
    assert_array_equal(supp, [24, 31, 20])

    # averaging tests
    sens = sensitivity_score(y_true, y_pred, average='macro')
    assert_array_almost_equal(sens, 0.60, 2)

    spec = specificity_score(y_true, y_pred, average='weighted')
    assert_array_almost_equal(spec, 0.80, 2)

    sens = sensitivity_score(y_true, y_pred, average='weighted')
    assert_array_almost_equal(sens, 0.53, 2)

    assert_raises(ValueError, sensitivity_score, y_true, y_pred,
                  average="samples")
    assert_raises(ValueError, specificity_score, y_true, y_pred,
                  average="samples")

    # same prediction but with and explicit label ordering
    sens, spec, supp = sensitivity_specificity_support(
        y_true, y_pred, labels=[0, 2, 1], average=None)
    assert_array_almost_equal(spec, [0.92, 0.55, 0.86], 2)
    assert_array_almost_equal(sens, [0.79, 0.90, 0.10], 2)
    assert_array_equal(supp, [24, 20, 31])
