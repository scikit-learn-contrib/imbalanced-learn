# coding: utf-8
"""Metrics to assess performance on classification task given class prediction

Functions named as ``*_score`` return a scalar value to maximize: the higher
the better

Function named as ``*_error`` or ``*_loss`` return a scalar value to minimize:
the lower the better
"""

from __future__ import division

import warnings
import logging

import numpy as np

from sklearn.metrics.classification import _check_targets, _prf_divide
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.fixes import bincount
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.sparsefuncs import count_nonzero

LOGGER = logging.getLogger(__name__)


def sensitivity_specificity_support(y_true,
                                    y_pred,
                                    labels=None,
                                    pos_label=1,
                                    average=None,
                                    warn_for=('sensitivity', 'specificity'),
                                    sample_weight=None):
    """Compute sensitivity, specificity, and support for each class

    The sensitivity is the ratio ``tp / (tp + fn)`` where ``tp`` is the number
    of true positives and ``fn`` the number of false negatives. The sensitivity
    quantifies the ability to avoid false negatives_[1].

    The specificity is the ratio ``tn / (tn + fp)`` where ``tn`` is the number
    of true negatives and ``fn`` the number of false negatives. The specificity
    quantifies the ability to avoid false positives_[1].

    The support is the number of occurrences of each class in ``y_true``.

    If ``pos_label is None`` and in binary classification, this function
    returns the average sensitivity and specificity if ``average``
    is one of ``'weighted'``.

    Parameters
    ----------
    y_true : ndarray, shape (n_samples, )
        Ground truth (correct) target values.

    y_pred : ndarray, shape (n_samples, )
        Estimated targets as returned by a classifier.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

    pos_label : str or int, optional (default=1)
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : str or None, optional (default=None)
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).
    warn_for : tuple or set, for internal use
        This determines which warnings will be made in the case that this
        function is being used to return only one of its metrics.

    sample_weight : ndarray, shape (n_samples, )
        Sample weights.

    Returns
    -------
    sensitivity : float (if ``average`` = None) or ndarray, \
        shape (n_unique_labels, )

    specificity : float (if ``average`` = None) or ndarray, \
        shape (n_unique_labels, )

    support : int (if ``average`` = None) or ndarray, \
        shape (n_unique_labels, )
        The number of occurrences of each label in ``y_true``.

    References
    ----------
    .. [1] `Wikipedia entry for the Sensitivity and specificity
           <https://en.wikipedia.org/wiki/Sensitivity_and_specificity>`_

    """
    average_options = (None, 'micro', 'macro', 'weighted', 'samples')
    if average not in average_options and average != 'binary':
        raise ValueError('average has to be one of ' + str(average_options))

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    present_labels = unique_labels(y_true, y_pred)

    if average == 'binary':
        if y_type == 'binary':
            if pos_label not in present_labels:
                if len(present_labels) < 2:
                    # Only negative labels
                    return (0., 0., 0)
                else:
                    raise ValueError("pos_label=%r is not a valid label: %r" %
                                     (pos_label, present_labels))
            labels = [pos_label]
        else:
            raise ValueError("Target is %s but average='binary'. Please "
                             "choose another average setting." % y_type)
    elif pos_label not in (None, 1):
        warnings.warn("Note that pos_label (set to %r) is ignored when "
                      "average != 'binary' (got %r). You may use "
                      "labels=[pos_label] to specify a single positive class."
                      % (pos_label, average), UserWarning)

    if labels is None:
        labels = present_labels
        n_labels = None
    else:
        n_labels = len(labels)
        labels = np.hstack(
            [labels, np.setdiff1d(
                present_labels, labels, assume_unique=True)])

    # Calculate tp_sum, pred_sum, true_sum ###

    if y_type.startswith('multilabel'):
        raise ValueError('imblearn does not support multilabel')
    elif average == 'samples':
        raise ValueError("Sample-based precision, recall, fscore is "
                         "not meaningful outside multilabel "
                         "classification. See the accuracy_score instead.")
    else:
        le = LabelEncoder()
        le.fit(labels)
        y_true = le.transform(y_true)
        y_pred = le.transform(y_pred)
        sorted_labels = le.classes_

        # labels are now from 0 to len(labels) - 1 -> use bincount
        tp = y_true == y_pred
        tp_bins = y_true[tp]
        if sample_weight is not None:
            tp_bins_weights = np.asarray(sample_weight)[tp]
        else:
            tp_bins_weights = None

        if len(tp_bins):
            tp_sum = bincount(
                tp_bins, weights=tp_bins_weights, minlength=len(labels))
        else:
            # Pathological case
            true_sum = pred_sum = tp_sum = np.zeros(len(labels))
        if len(y_pred):
            pred_sum = bincount(
                y_pred, weights=sample_weight, minlength=len(labels))
        if len(y_true):
            true_sum = bincount(
                y_true, weights=sample_weight, minlength=len(labels))

        # Compute the true negative
        tn_sum = y_true.size - (pred_sum + true_sum - tp_sum)

        # Retain only selected labels
        indices = np.searchsorted(sorted_labels, labels[:n_labels])
        tp_sum = tp_sum[indices]
        true_sum = true_sum[indices]
        pred_sum = pred_sum[indices]
        tn_sum = tn_sum[indices]

    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        pred_sum = np.array([pred_sum.sum()])
        true_sum = np.array([true_sum.sum()])
        tn_sum = np.array([tn_sum.sum()])

    # Finally, we have all our sufficient statistics. Divide! #

    with np.errstate(divide='ignore', invalid='ignore'):
        # Divide, and on zero-division, set scores to 0 and warn:

        # Oddly, we may get an "invalid" rather than a "divide" error
        # here.
        specificity = _prf_divide(tn_sum, tn_sum + pred_sum - tp_sum,
                                  'specificity', 'predicted', average,
                                  warn_for)
        sensitivity = _prf_divide(tp_sum, true_sum, 'sensitivity', 'true',
                                  average, warn_for)

    # Average the results

    if average == 'weighted':
        weights = true_sum
        if weights.sum() == 0:
            return 0, 0, None
    elif average == 'samples':
        weights = sample_weight
    else:
        weights = None

    if average is not None:
        assert average != 'binary' or len(specificity) == 1
        specificity = np.average(specificity, weights=weights)
        sensitivity = np.average(sensitivity, weights=weights)
        true_sum = None  # return no support

    return sensitivity, specificity, true_sum


def sensitivity_score(y_true,
                      y_pred,
                      labels=None,
                      pos_label=1,
                      average='binary',
                      sample_weight=None):
    """Compute the sensitivity

    The sensitivity is the ratio ``tp / (tp + fn)`` where ``tp`` is the number
    of true positives and ``fn`` the number of false negatives. The sensitivity
    quantifies the ability to avoid false negatives.

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    y_true : ndarray, shape (n_samples, )
        Ground truth (correct) target values.

    y_pred : ndarray, shape (n_samples, )
        Estimated targets as returned by a classifier.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average.

    pos_label : str or int, optional (default=1)
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : str or None, optional (default=None)
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    warn_for : tuple or set, for internal use
        This determines which warnings will be made in the case that this
        function is being used to return only one of its metrics.

    sample_weight : ndarray, shape (n_samples, )
        Sample weights.

    Returns
    -------
    specificity : float (if ``average`` = None) or ndarray, \
        shape (n_unique_labels, )

    """
    s, _, _ = sensitivity_specificity_support(
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=('sensitivity', ),
        sample_weight=sample_weight)

    return s


def specificity_score(y_true,
                      y_pred,
                      labels=None,
                      pos_label=1,
                      average='binary',
                      sample_weight=None):
    """Compute the specificity

    The specificity is the ratio ``tp / (tp + fn)`` where ``tp`` is the number
    of true positives and ``fn`` the number of false negatives. The specificity
    is intuitively the ability of the classifier to find all the positive
    samples.

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    y_true : ndarray, shape (n_samples, )
        Ground truth (correct) target values.

    y_pred : ndarray, shape (n_samples, )
        Estimated targets as returned by a classifier.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average.

    pos_label : str or int, optional (default=1)
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : str or None, optional (default=None)
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    warn_for : tuple or set, for internal use
        This determines which warnings will be made in the case that this
        function is being used to return only one of its metrics.

    sample_weight : ndarray, shape (n_samples, )
        Sample weights.

    Returns
    -------
    specificity : float (if ``average`` = None) or ndarray, \
        shape (n_unique_labels, )

    """
    _, s, _ = sensitivity_specificity_support(
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=('specificity', ),
        sample_weight=sample_weight)

    return s


def geometric_mean_score(y_true,
                         y_pred,
                         labels=None,
                         pos_label=1,
                         average='binary',
                         sample_weight=None):
    """Compute the geometric mean

    The geometric mean is the squared root of the product of the sensitivity
    and specificity. This measure tries to maximize the accuracy on each
    of the two classes while keeping these accuracies balanced.

    The specificity is the ratio ``tp / (tp + fn)`` where ``tp`` is the number
    of true positives and ``fn`` the number of false negatives. The specificity
    is intuitively the ability of the classifier to find all the positive
    samples.

    The best value is 1 and the worst value is 0.

    Parameters
    ----------
    y_true : ndarray, shape (n_samples, )
        Ground truth (correct) target values.

    y_pred : ndarray, shape (n_samples, )
        Estimated targets as returned by a classifier.

    labels : list, optional
        The set of labels to include when ``average != 'binary'``, and their
        order if ``average is None``. Labels present in the data can be
        excluded, for example to calculate a multiclass average ignoring a
        majority negative class, while labels not present in the data will
        result in 0 components in a macro average.

    pos_label : str or int, optional (default=1)
        The class to report if ``average='binary'`` and the data is binary.
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : str or None, optional (default=None)
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'micro'``:
            Calculate metrics globally by counting the total true positives,
            false negatives and false positives.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance; it can result in an
            F-score that is not between precision and recall.
        ``'samples'``:
            Calculate metrics for each instance, and find their average (only
            meaningful for multilabel classification where this differs from
            :func:`accuracy_score`).

    warn_for : tuple or set, for internal use
        This determines which warnings will be made in the case that this
        function is being used to return only one of its metrics.

    sample_weight : ndarray, shape (n_samples, )
        Sample weights.

    Returns
    -------
    geometric_mean : float (if ``average`` = None) or ndarray, \
        shape (n_unique_labels, )

    References
    ----------
    .. [1] Kubat, M. and Matwin, S. "Addressing the curse of
       imbalanced training sets: one-sided selection" ICML (1997)

    .. [2] Barandela, R., Sánchez, J. S., Garcıa, V., & Rangel, E. "Strategies
       for learning in class imbalance problems", Pattern Recognition,
       36(3), (2003), pp 849-851.

    """
    sen, spe, _ = sensitivity_specificity_support(
        y_true,
        y_pred,
        labels=labels,
        pos_label=pos_label,
        average=average,
        warn_for=('specificity', 'specificity'),
        sample_weight=sample_weight)

    LOGGER.debug('The sensitivity and specificity are : %s - %s' % (sen, spe))

    return np.sqrt(sen * spe)


def indexed_balanced_accuracy_score(score_func,
                                    y_true,
                                    y_pred,
                                    alpha=0.1,
                                    squared=True,
                                    **kwargs):
    """ Compute the indexed balanced accuracy of a scoring function

    The indexed balanced accuracy (IBA) tends to weight a scoring function
    to take into account the imbalancing of the data.

    Parameters
    ----------
    score_func : callable,
        Score function (or loss function) with signature
        ``score_func(y, y_pred, **kwargs)``.

    y_true : ndarray, shape (n_samples, )
        Ground truth (correct) target values.

    y_pred : ndarray, shape (n_samples, )
        Estimated targets as returned by a classifier.

    alpha : float, optional (default=0.1)
        Weighting factor.

    squared : bool, optional (default=True)
        If ``squared`` is True, then the metric computed will be squared
        before to be weighted.

    **kwargs : additional arguments
        Additional parameters to be passed to score_func.

    Returns
    -------
    iba : float (if ``average`` = None) or ndarray, \
        shape (n_unique_labels, )

    References
    ----------
    .. [1] Garcia, V. and Mollineda, R.A. and Sanchez, J.S. "Theoretical
       analysis of a performance measure for imbalanced data" ICPR (2010)
    """

    score = score_func(**kwargs)

    if squared:
        score = np.power(score, 2)

    # Pop the arguments to have the proper average, etc. for the
    # sensitivity and specificity
    labels = kwargs.get('labels', None)
    pos_label = kwargs.get('pos_label', 1)
    average = kwargs.get('average', 'binary')
    sample_weight = kwargs.get('sample_weight', None)

    # Compute the sensitivity and specificity
    sen = sensitivity_score(y_true, y_pred, labels, pos_label, average,
                            sample_weight)
    spe = specificity_score(y_true, y_pred, labels, pos_label, average,
                            sample_weight)

    # Compute the dominance
    dom = sen - spe

    return (1. + alpha * dom) * score
