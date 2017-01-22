"""Testing the metric for classification with imbalanced dataset"""

from __future__ import division, print_function

from functools import partial

import numpy as np

from numpy.testing import (assert_allclose, assert_array_equal,
                           assert_no_warnings, assert_equal,
                           assert_raises)
from sklearn.utils.testing import assert_warns_message, ignore_warnings

from sklearn import datasets
from sklearn import svm

from sklearn.preprocessing import label_binarize
from sklearn.utils.fixes import np_version
from sklearn.utils.testing import assert_not_equal, assert_raise_message
from sklearn.utils.validation import check_random_state
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, cohen_kappa_score,
                             jaccard_similarity_score, precision_score,
                             recall_score, roc_auc_score)

from imblearn.metrics import sensitivity_specificity_support
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from imblearn.metrics import geometric_mean_score
from imblearn.metrics import make_index_balanced_accuracy
from imblearn.metrics import classification_report_imbalanced

RND_SEED = 42
R_TOL = 1e-2

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


def test_sensitivity_specificity_score_binary():
    """Test Sensitivity Specificity for binary classification task"""
    y_true, y_pred, _ = make_prediction(binary=True)

    # detailed measures for each class
    sen, spe, sup = sensitivity_specificity_support(
        y_true, y_pred, average=None)
    assert_allclose(sen, [0.88, 0.68], rtol=R_TOL)
    assert_allclose(spe, [0.68, 0.88], rtol=R_TOL)
    assert_array_equal(sup, [25, 25])

    # individual scoring function that can be used for grid search: in the
    # binary class case the score is the value of the measure for the positive
    # class (e.g. label == 1). This is deprecated for average != 'binary'.
    for kwargs, my_assert in [({}, assert_no_warnings), ({
            'average': 'binary'
    }, assert_no_warnings)]:
        sen = my_assert(sensitivity_score, y_true, y_pred, **kwargs)
        assert_allclose(sen, 0.68, rtol=R_TOL)

        spe = my_assert(specificity_score, y_true, y_pred, **kwargs)
        assert_allclose(spe, 0.88, rtol=R_TOL)


def test_sensitivity_specificity_f_binary_single_class():
    """Test sensitivity and specificity behave with a single positive or
    negative class"""
    # Such a case may occur with non-stratified cross-validation
    assert_equal(1., sensitivity_score([1, 1], [1, 1]))
    assert_equal(0., specificity_score([1, 1], [1, 1]))

    assert_equal(0., sensitivity_score([-1, -1], [-1, -1]))
    assert_equal(0., specificity_score([-1, -1], [-1, -1]))


@ignore_warnings
def test_sensitivity_specificity_extra_labels():
    """Test handling of explicit additional (not in input) labels to SS"""
    y_true = [1, 3, 3, 2]
    y_pred = [1, 1, 3, 2]

    # No average: zeros in array
    actual = specificity_score(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None)
    assert_allclose([1., 0.67, 1., 1., 1.], actual, rtol=R_TOL)

    # Macro average is changed
    actual = specificity_score(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], average='macro')
    assert_allclose(np.mean([1., 0.67, 1., 1., 1.]), actual, rtol=R_TOL)

    # Check for micro
    actual = specificity_score(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], average='micro')
    assert_allclose(15. / 16., actual, rtol=R_TOL)

    # Check for weighted
    actual = specificity_score(
        y_true, y_pred, labels=[0, 1, 2, 3, 4], average='macro')
    assert_allclose(np.mean([1., 0.67, 1., 1., 1.]), actual, rtol=R_TOL)


@ignore_warnings
def test_sensitivity_specificity_ignored_labels():
    """Test a subset of labels may be requested for SS"""
    y_true = [1, 1, 2, 3]
    y_pred = [1, 3, 3, 3]

    specificity_13 = partial(specificity_score, y_true, y_pred, labels=[1, 3])
    specificity_all = partial(specificity_score, y_true, y_pred, labels=None)

    assert_allclose([1., 0.33], specificity_13(average=None), rtol=R_TOL)
    assert_allclose(
        np.mean([1., 0.33]), specificity_13(average='macro'), rtol=R_TOL)
    assert_allclose(
        np.average(
            [1., .33], weights=[2., 1.]),
        specificity_13(average='weighted'),
        rtol=R_TOL)
    assert_allclose(3. / (3. + 2.), specificity_13(average='micro'),
                    rtol=R_TOL)

    # ensure the above were meaningful tests:
    for average in ['macro', 'weighted', 'micro']:
        assert_not_equal(
            specificity_13(average=average), specificity_all(average=average))


def test_sensitivity_specificity_error_multilabels():
    """Test either if an error is raised when the input are multilabels"""
    y_true = [1, 3, 3, 2]
    y_pred = [1, 1, 3, 2]
    y_true_bin = label_binarize(y_true, classes=np.arange(5))
    y_pred_bin = label_binarize(y_pred, classes=np.arange(5))

    assert_raises(ValueError, sensitivity_score, y_true_bin, y_pred_bin)


@ignore_warnings
def test_sensitivity_specificity_support_errors():
    """Test either if an error is raised depending on parameters"""
    y_true, y_pred, _ = make_prediction(binary=True)

    # Bad pos_label
    assert_raises(
        ValueError,
        sensitivity_specificity_support,
        y_true,
        y_pred,
        pos_label=2,
        average='binary')

    # Bad average option
    assert_raises(
        ValueError,
        sensitivity_specificity_support, [0, 1, 2], [1, 2, 0],
        average='mega')


def test_sensitivity_specificity_unused_pos_label():
    """Check warning that pos_label unused when set to non-default value
    # but average != 'binary'; even if data is binary"""
    assert_warns_message(
        UserWarning,
        "Note that pos_label (set to 2) is "
        "ignored when average != 'binary' (got 'macro'). You "
        "may use labels=[pos_label] to specify a single "
        "positive class.",
        sensitivity_specificity_support, [1, 2, 1], [1, 2, 2],
        pos_label=2,
        average='macro')


def test_geometric_mean_support_binary():
    """Test the geometric mean for binary classification task"""
    y_true, y_pred, _ = make_prediction(binary=True)

    # compute the geometric mean for the binary problem
    geo_mean = geometric_mean_score(y_true, y_pred)

    assert_allclose(geo_mean, 0.77, rtol=R_TOL)


def test_geometric_mean_multiclass():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    assert_allclose(geometric_mean_score(y_true, y_pred), 1.0, rtol=R_TOL)

    y_true = [0, 0, 0, 0]
    y_pred = [1, 1, 1, 1]
    assert_allclose(geometric_mean_score(y_true, y_pred), 0.0, rtol=R_TOL)

    cor = 0.001
    y_true = [0, 0, 0, 0]
    y_pred = [0, 0, 0, 0]
    assert_allclose(geometric_mean_score(y_true, y_pred, correction=cor),
                    1.0, rtol=R_TOL)

    y_true = [0, 0, 0, 0]
    y_pred = [1, 1, 1, 1]
    assert_allclose(geometric_mean_score(y_true, y_pred, correction=cor),
                    cor, rtol=R_TOL)

    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 0]
    assert_allclose(geometric_mean_score(y_true, y_pred, correction=cor),
                    0.5, rtol=R_TOL)

    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]
    assert_allclose(geometric_mean_score(y_true, y_pred, correction=cor),
                    (1*cor*cor)**(1.0/3.0), rtol=R_TOL)

    y_true = [0, 1, 2, 3, 4, 5]
    y_pred = [0, 1, 2, 3, 4, 5]
    assert_allclose(geometric_mean_score(y_true, y_pred, correction=cor),
                    1, rtol=R_TOL)

    y_true = [0, 1, 1, 1, 1, 0]
    y_pred = [0, 0, 1, 1, 1, 1]
    assert_allclose(geometric_mean_score(y_true, y_pred, correction=cor),
                    (0.5*0.75)**0.5, rtol=R_TOL)

    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 2, 1, 0, 0, 1]
    assert_allclose(geometric_mean_score(y_true, y_pred, average='macro'),
                    0.47140452079103168, rtol=R_TOL)
    assert_allclose(geometric_mean_score(y_true, y_pred, average='micro'),
                    0.47140452079103168, rtol=R_TOL)
    assert_allclose(geometric_mean_score(y_true, y_pred,
                                         average='weighted'),
                    0.47140452079103168, rtol=R_TOL)
    assert_allclose(geometric_mean_score(y_true, y_pred, average=None),
                    [0.8660254, 0.0, 0.0], rtol=R_TOL)

    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 1, 0, 0, 1]
    assert_allclose(geometric_mean_score(y_true, y_pred, labels=[0, 1]),
                    0.70710678118654752, rtol=R_TOL)
    assert_allclose(geometric_mean_score(y_true, y_pred, labels=[0, 1],
                                         sample_weight=[1, 2, 1, 1, 2, 1]),
                    0.70710678118654752, rtol=R_TOL)
    assert_allclose(geometric_mean_score(y_true, y_pred, labels=[0, 1],
                                         sample_weight=[1, 2, 1, 1, 2, 1],
                                         average='weighted'),
                    0.3333333333, rtol=R_TOL)

    y_true, y_pred, _ = make_prediction(binary=False)

    geo_mean = geometric_mean_score(y_true, y_pred)
    assert_allclose(geo_mean, 0.41, rtol=R_TOL)

    # Compute the geometric mean for each of the classes
    geo_mean = geometric_mean_score(y_true, y_pred, average=None)
    assert_allclose(geo_mean, [0.85, 0.29, 0.7], rtol=R_TOL)

    # average tests
    geo_mean = geometric_mean_score(y_true, y_pred, average='macro')
    assert_allclose(geo_mean, 0.68, rtol=R_TOL)

    geo_mean = geometric_mean_score(y_true, y_pred, average='weighted')
    assert_allclose(geo_mean, 0.65, rtol=R_TOL)


def test_iba_geo_mean_binary():
    """Test to test the iba using the geometric mean"""
    y_true, y_pred, _ = make_prediction(binary=True)

    iba_gmean = make_index_balanced_accuracy(
        alpha=0.5, squared=True)(geometric_mean_score)
    iba = iba_gmean(y_true, y_pred)

    assert_allclose(iba, 0.5948, rtol=R_TOL)


def _format_report(report):
    """Private function to reformat the report for testing"""

    return ' '.join(report.split())


def test_classification_report_imbalanced_multiclass():
    """Test classification report for multiclass problem"""
    iris = datasets.load_iris()
    y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)

    # print classification report with class names
    expected_report = ('pre rec spe f1 geo iba sup setosa 0.83 0.79 0.92 '
                       '0.81 0.86 0.74 24 versicolor 0.33 0.10 0.86 0.15 '
                       '0.44 0.19 31 virginica 0.42 0.90 0.55 0.57 0.63 '
                       '0.37 20 avg / total 0.51 0.53 0.80 0.47 0.62 0.41 75')

    report = classification_report_imbalanced(
        y_true,
        y_pred,
        labels=np.arange(len(iris.target_names)),
        target_names=iris.target_names)
    assert_equal(_format_report(report), expected_report)
    # print classification report with label detection
    expected_report = ('pre rec spe f1 geo iba sup 0 0.83 0.79 0.92 0.81 '
                       '0.86 0.74 24 1 0.33 0.10 0.86 0.15 0.44 0.19 31 2 '
                       '0.42 0.90 0.55 0.57 0.63 0.37 20 avg / total 0.51 '
                       '0.53 0.80 0.47 0.62 0.41 75')

    report = classification_report_imbalanced(y_true, y_pred)
    assert_equal(_format_report(report), expected_report)


def test_classification_report_imbalanced_multiclass_with_digits():
    """Test performance report with added digits in floating point values"""
    iris = datasets.load_iris()
    y_true, y_pred, _ = make_prediction(dataset=iris, binary=False)

    # print classification report with class names
    expected_report = ('pre rec spe f1 geo iba sup setosa 0.82609 0.79167 '
                       '0.92157 0.80851 0.86409 0.74085 24 versicolor '
                       '0.33333 0.09677 0.86364 0.15000 0.43809 0.18727 31 '
                       'virginica 0.41860 0.90000 0.54545 0.57143 0.62645 '
                       '0.37208 20 avg / total 0.51375 0.53333 0.79733 '
                       '0.47310 0.62464 0.41370 75')
    report = classification_report_imbalanced(
        y_true,
        y_pred,
        labels=np.arange(len(iris.target_names)),
        target_names=iris.target_names,
        digits=5)
    assert_equal(_format_report(report), expected_report)
    # print classification report with label detection
    expected_report = ('pre rec spe f1 geo iba sup 0 0.83 0.79 0.92 0.81 '
                       '0.86 0.74 24 1 0.33 0.10 0.86 0.15 0.44 0.19 31 2 '
                       '0.42 0.90 0.55 0.57 0.63 0.37 20 avg / total 0.51 '
                       '0.53 0.80 0.47 0.62 0.41 75')
    report = classification_report_imbalanced(y_true, y_pred)
    assert_equal(_format_report(report), expected_report)


def test_classification_report_imbalanced_multiclass_with_string_label():
    """Test the report with string label"""
    y_true, y_pred, _ = make_prediction(binary=False)

    y_true = np.array(["blue", "green", "red"])[y_true]
    y_pred = np.array(["blue", "green", "red"])[y_pred]

    expected_report = ('pre rec spe f1 geo iba sup blue 0.83 0.79 0.92 '
                       '0.81 0.86 0.74 24 green 0.33 0.10 0.86 0.15 0.44 '
                       '0.19 31 red 0.42 0.90 0.55 0.57 0.63 0.37 20 '
                       'avg / total 0.51 0.53 0.80 0.47 0.62 0.41 75')
    report = classification_report_imbalanced(y_true, y_pred)
    assert_equal(_format_report(report), expected_report)

    expected_report = ('pre rec spe f1 geo iba sup a 0.83 0.79 0.92 0.81 '
                       '0.86 0.74 24 b 0.33 0.10 0.86 0.15 0.44 0.19 31 '
                       'c 0.42 0.90 0.55 0.57 0.63 0.37 20 avg / total '
                       '0.51 0.53 0.80 0.47 0.62 0.41 75')
    report = classification_report_imbalanced(
        y_true, y_pred, target_names=["a", "b", "c"])
    assert_equal(_format_report(report), expected_report)


def test_classification_report_imbalanced_multiclass_with_unicode_label():
    """Test classification report with unicode label"""
    y_true, y_pred, _ = make_prediction(binary=False)

    labels = np.array([u"blue\xa2", u"green\xa2", u"red\xa2"])
    y_true = labels[y_true]
    y_pred = labels[y_pred]

    expected_report = (u'pre rec spe f1 geo iba sup blue\xa2 0.83 0.79 '
                       u'0.92 0.81 0.86 0.74 24 green\xa2 0.33 0.10 0.86 '
                       u'0.15 0.44 0.19 31 red\xa2 0.42 0.90 0.55 0.57 0.63 '
                       u'0.37 20 avg / total 0.51 0.53 0.80 0.47 0.62 0.41 75')
    if np_version[:3] < (1, 7, 0):
        expected_message = ("NumPy < 1.7.0 does not implement"
                            " searchsorted on unicode data correctly.")
        assert_raise_message(RuntimeError, expected_message,
                             classification_report_imbalanced, y_true, y_pred)
    else:
        report = classification_report_imbalanced(y_true, y_pred)
        assert_equal(_format_report(report), expected_report)


def test_classification_report_imbalanced_multiclass_with_long_string_label():
    """Test classification report with long string label"""
    y_true, y_pred, _ = make_prediction(binary=False)

    labels = np.array(["blue", "green" * 5, "red"])
    y_true = labels[y_true]
    y_pred = labels[y_pred]

    expected_report = ('pre rec spe f1 geo iba sup blue 0.83 0.79 0.92 0.81 '
                       '0.86 0.74 24 greengreengreengreengreen 0.33 0.10 '
                       '0.86 0.15 0.44 0.19 31 red 0.42 0.90 0.55 0.57 0.63 '
                       '0.37 20 avg / total 0.51 0.53 0.80 0.47 0.62 0.41 75')

    report = classification_report_imbalanced(y_true, y_pred)
    assert_equal(_format_report(report), expected_report)


def test_iba_sklearn_metrics():
    """Test the compatibility of sklearn metrics within IBA"""
    y_true, y_pred, _ = make_prediction(binary=True)

    acc = make_index_balanced_accuracy(alpha=0.5, squared=True)(
        accuracy_score)
    score = acc(y_true, y_pred)
    assert_equal(score, 0.54756)

    jss = make_index_balanced_accuracy(alpha=0.5, squared=True)(
        jaccard_similarity_score)
    score = jss(y_true, y_pred)
    assert_equal(score, 0.54756)

    pre = make_index_balanced_accuracy(alpha=0.5, squared=True)(
        precision_score)
    score = pre(y_true, y_pred)
    assert_equal(score, 0.65025)

    rec = make_index_balanced_accuracy(alpha=0.5, squared=True)(
        recall_score)
    score = rec(y_true, y_pred)
    assert_equal(score, 0.41616000000000009)


def test_iba_error_y_score_prob():
    """Test if an error is raised when a scoring metric take over parameters
    than y_pred"""
    y_true, y_pred, _ = make_prediction(binary=True)

    aps = make_index_balanced_accuracy(alpha=0.5, squared=True)(
        average_precision_score)
    assert_raises(AttributeError, aps, y_true, y_pred)

    brier = make_index_balanced_accuracy(alpha=0.5, squared=True)(
        brier_score_loss)
    assert_raises(AttributeError, brier, y_true, y_pred)

    kappa = make_index_balanced_accuracy(alpha=0.5, squared=True)(
        cohen_kappa_score)
    assert_raises(AttributeError, kappa, y_true, y_pred)

    ras = make_index_balanced_accuracy(alpha=0.5, squared=True)(
        roc_auc_score)
    assert_raises(AttributeError, ras, y_true, y_pred)
