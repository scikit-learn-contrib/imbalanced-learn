"""Testing the metric for classification with imbalanced dataset"""

from __future__ import division, print_function

import numpy as np

from numpy.testing import (assert_array_almost_equal, assert_array_equal,
                           assert_no_warnings, assert_equal,
                           assert_almost_equal, assert_raises)
from sklearn.utils.testing import assert_warns_message, ignore_warnings

from sklearn import datasets
from sklearn import svm

from sklearn.utils.validation import check_random_state

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

def test_precision_recall_f1_score_binary():
    # Test Precision Recall and F1 Score for binary classification task
    y_true, y_pred, _ = make_prediction(binary=True)

    # detailed measures for each class
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, average=None)
    assert_array_almost_equal(p, [0.73, 0.85], 2)
    assert_array_almost_equal(r, [0.88, 0.68], 2)
    assert_array_almost_equal(f, [0.80, 0.76], 2)
    assert_array_equal(s, [25, 25])

    # individual scoring function that can be used for grid search: in the
    # binary class case the score is the value of the measure for the positive
    # class (e.g. label == 1). This is deprecated for average != 'binary'.
    for kwargs, my_assert in [({}, assert_no_warnings),
                              ({'average': 'binary'}, assert_no_warnings)]:
        ps = my_assert(precision_score, y_true, y_pred, **kwargs)
        assert_array_almost_equal(ps, 0.85, 2)

        rs = my_assert(recall_score, y_true, y_pred, **kwargs)
        assert_array_almost_equal(rs, 0.68, 2)

        fs = my_assert(f1_score, y_true, y_pred, **kwargs)
        assert_array_almost_equal(fs, 0.76, 2)

        assert_almost_equal(my_assert(fbeta_score, y_true, y_pred, beta=2,
                                      **kwargs),
                            (1 + 2 ** 2) * ps * rs / (2 ** 2 * ps + rs), 2)


def test_precision_recall_f_binary_single_class():
    # Test precision, recall and F1 score behave with a single positive or
    # negative class
    # Such a case may occur with non-stratified cross-validation
    assert_equal(1., precision_score([1, 1], [1, 1]))
    assert_equal(1., recall_score([1, 1], [1, 1]))
    assert_equal(1., f1_score([1, 1], [1, 1]))

    assert_equal(0., precision_score([-1, -1], [-1, -1]))
    assert_equal(0., recall_score([-1, -1], [-1, -1]))
    assert_equal(0., f1_score([-1, -1], [-1, -1]))


@ignore_warnings
def test_precision_recall_f_extra_labels():
    # Test handling of explicit additional (not in input) labels to PRF
    y_true = [1, 3, 3, 2]
    y_pred = [1, 1, 3, 2]
    y_true_bin = label_binarize(y_true, classes=np.arange(5))
    y_pred_bin = label_binarize(y_pred, classes=np.arange(5))
    data = [(y_true, y_pred),
            (y_true_bin, y_pred_bin)]

    for i, (y_true, y_pred) in enumerate(data):
        # No average: zeros in array
        actual = recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4],
                              average=None)
        assert_array_almost_equal([0., 1., 1., .5, 0.], actual)

        # Macro average is changed
        actual = recall_score(y_true, y_pred, labels=[0, 1, 2, 3, 4],
                              average='macro')
        assert_array_almost_equal(np.mean([0., 1., 1., .5, 0.]), actual)

        # No effect otheriwse
        for average in ['micro', 'weighted', 'samples']:
            if average == 'samples' and i == 0:
                continue
            assert_almost_equal(recall_score(y_true, y_pred,
                                             labels=[0, 1, 2, 3, 4],
                                             average=average),
                                recall_score(y_true, y_pred, labels=None,
                                             average=average))

    # Error when introducing invalid label in multilabel case
    # (although it would only affect performance if average='macro'/None)
    for average in [None, 'macro', 'micro', 'samples']:
        assert_raises(ValueError, recall_score, y_true_bin, y_pred_bin,
                      labels=np.arange(6), average=average)
        assert_raises(ValueError, recall_score, y_true_bin, y_pred_bin,
                      labels=np.arange(-1, 4), average=average)


@ignore_warnings
def test_precision_recall_f_ignored_labels():
    # Test a subset of labels may be requested for PRF
    y_true = [1, 1, 2, 3]
    y_pred = [1, 3, 3, 3]
    y_true_bin = label_binarize(y_true, classes=np.arange(5))
    y_pred_bin = label_binarize(y_pred, classes=np.arange(5))
    data = [(y_true, y_pred),
            (y_true_bin, y_pred_bin)]

    for i, (y_true, y_pred) in enumerate(data):
        recall_13 = partial(recall_score, y_true, y_pred, labels=[1, 3])
        recall_all = partial(recall_score, y_true, y_pred, labels=None)

        assert_array_almost_equal([.5, 1.], recall_13(average=None))
        assert_almost_equal((.5 + 1.) / 2, recall_13(average='macro'))
        assert_almost_equal((.5 * 2 + 1. * 1) / 3,
                            recall_13(average='weighted'))
        assert_almost_equal(2. / 3, recall_13(average='micro'))

        # ensure the above were meaningful tests:
        for average in ['macro', 'weighted', 'micro']:
            assert_not_equal(recall_13(average=average),
                             recall_all(average=average))


@ignore_warnings
def test_precision_recall_fscore_support_errors():
    y_true, y_pred, _ = make_prediction(binary=True)

    # Bad beta
    assert_raises(ValueError, precision_recall_fscore_support,
                  y_true, y_pred, beta=0.0)

    # Bad pos_label
    assert_raises(ValueError, precision_recall_fscore_support,
                  y_true, y_pred, pos_label=2, average='binary')

    # Bad average option
    assert_raises(ValueError, precision_recall_fscore_support,
                  [0, 1, 2], [1, 2, 0], average='mega')


def test_precision_recall_f_unused_pos_label():
    # Check warning that pos_label unused when set to non-default value
    # but average != 'binary'; even if data is binary.
    assert_warns_message(UserWarning,
                         "Note that pos_label (set to 2) is "
                         "ignored when average != 'binary' (got 'macro'). You "
                         "may use labels=[pos_label] to specify a single "
                         "positive class.", precision_recall_fscore_support,
                         [1, 2, 1], [1, 2, 2], pos_label=2, average='macro')
