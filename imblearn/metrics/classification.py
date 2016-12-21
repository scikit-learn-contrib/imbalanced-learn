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

LOGGER = logging.getLogger(__name__)


def sensitivity_specificity_support(y_true, y_pred, labels=None,
                                    pos_label=1, average=None,
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
    is one of ``'micro'`` or 'weighted'``.

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
        If the data are multiclass or multilabel, this will be ignored;
        setting ``labels=[pos_label]`` and ``average != 'binary'`` will report
        scores for that label only.

    average : str or None, optional (default=None)
        If ``None``, the scores for each class are returned. Otherwise, this
        determines the type of averaging performed on the data:

        ``'binary'``:
            Only report results for the class specified by ``pos_label``.
            This is applicable only if targets (``y_{true,pred}``) are binary.
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance.

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

    average_options = (None, 'micro', 'macro', 'weighted')
    if average not in average_options and average != 'binary':
        raise ValueError('average has to be one of ' +
                         str(average_options))

    y_type, y_true, y_pred = _check_targets(y_true, y_pred)
    present_labels = unique_labels(y_true, y_pred)

    LOGGER.debug('The labels in the prediction and ground-truth are %s',
                 present_labels)

    # We do not support multilabel for the moment
    if y_type.startswith('multilabel'):
        raise ValueError('Multilabel are not supported.')

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
        labels = np.hstack([labels, np.setdiff1d(present_labels, labels,
                                                 assume_unique=True)])

    le = LabelEncoder()
    le.fit(labels)
    y_true = le.transform(y_true)
    y_pred = le.transform(y_pred)
    sorted_labels = le.classes_

    LOGGER.debug(y_true)
    LOGGER.debug(y_pred)
    LOGGER.debug(sorted_labels)

    LOGGER.debug('The number of labels is %s' % n_labels)

    # In a leave out strategy and for each label, compute:
    # TP, TN, FP, FN
    # These list contain an array in which each sample is labeled as
    # TP, TN, FP, FN
    list_tp = [np.bitwise_and((y_true == label), (y_pred == label))
               for label in sorted_labels]
    list_tn = [np.bitwise_and((y_true != label), (y_pred != label))
               for label in sorted_labels]
    list_fp = [np.bitwise_and((y_true != label), (y_pred == label))
               for label in sorted_labels]
    list_fn = [np.bitwise_and((y_true == label), (y_pred != label))
               for label in sorted_labels]

    # Compute the sum for each type
    # We keep only the counting corresponding to True values
    # We are using bincount since it allows to weight the samples
    tp_sum = np.array([bincount(tp, weights=sample_weight,
                                minlength=2)[-1]
                       for tp in list_tp])
    tn_sum = np.array([bincount(tn, weights=sample_weight,
                                minlength=2)[-1]
                       for tn in list_tn])
    fp_sum = np.array([bincount(fp, weights=sample_weight,
                                minlength=2)[-1]
                       for fp in list_fp])
    fn_sum = np.array([bincount(fn, weights=sample_weight,
                                minlength=2)[-1]
                       for fn in list_fn])

    # Retain only selected labels
    indices = np.searchsorted(sorted_labels, labels[:n_labels])
    # For support, we can count the number of occurrences of each label
    support = np.array(bincount(y_true, weights=sample_weight,
                                minlength=len(labels)))
    # Sort the support
    support = support[indices]


    LOGGER.debug('The indices which are retained are %s' % indices)

    LOGGER.debug('TP: %s' % tp_sum)
    LOGGER.debug('TN: %s' % tn_sum)
    LOGGER.debug('FP: %s' % fp_sum)
    LOGGER.debug('FN: %s' % fn_sum)

    tp_sum = tp_sum[indices]
    tn_sum = tn_sum[indices]
    fp_sum = fp_sum[indices]
    fn_sum = fn_sum[indices]

    if average == 'micro':
        tp_sum = np.array([tp_sum.sum()])
        tn_sum = np.array([tn_sum.sum()])
        fp_sum = np.array([fp_sum.sum()])
        fn_sum = np.array([fn_sum.sum()])

    LOGGER.debug('Did we do the average micro %s' % tp_sum)

    LOGGER.debug('Computed the necessary stats for the sensitivity and'
                 ' specificity')

    # Compute the sensitivity and specificity
    sensitivity = [_prf_divide(tp, tp + fn, 'sensitivity', 'tp + fn', average,
                               warn_for) for tp, fn in zip(tp_sum, fn_sum)]
    specificity = [_prf_divide(tn, tn + fp, 'specificity', 'tn + fp', average,
                               warn_for) for tn, fp in zip(tn_sum, fp_sum)]

    LOGGER.debug('Sensitivity = %s - Specificity = %s' % (sensitivity,
                                                          specificity))

    LOGGER.debug('Computed the sensitivity and specificity for each class')
    LOGGER.debug('The lengths of those two metrics are: %s - %s',
                 len(sensitivity), len(specificity))

    # If we need to weight the results
    if average == 'weighted':
        weights = support
        if weights.sum() == 0:
            return 0, 0, None
    else:
        weights = None

    if average is not None:
        assert average != 'binary' or len(sensitivity) == 1
        sensitivity = np.average(sensitivity, weights=weights)
        specificity = np.average(specificity, weights=weights)
        support = None

    return sensitivity, specificity, support


def sensitivity_score(y_true, y_pred, labels=None, pos_label=1,
                      average='binary', sample_weight=None):
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
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

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
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance.

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
    s, _, _ = sensitivity_specificity_support(y_true, y_pred,
                                              labels=labels,
                                              pos_label=pos_label,
                                              average=average,
                                              warn_for=('specificity',),
                                              sample_weight=sample_weight)

    return s


def specificity_score(y_true, y_pred, labels=None, pos_label=1,
                      average='binary', sample_weight=None):
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
        result in 0 components in a macro average. For multilabel targets,
        labels are column indices. By default, all labels in ``y_true`` and
        ``y_pred`` are used in sorted order.

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
        ``'macro'``:
            Calculate metrics for each label, and find their unweighted
            mean.  This does not take label imbalance into account.
        ``'weighted'``:
            Calculate metrics for each label, and find their average, weighted
            by support (the number of true instances for each label). This
            alters 'macro' to account for label imbalance.

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
    _, s, _ = sensitivity_specificity_support(y_true, y_pred,
                                              labels=labels,
                                              pos_label=pos_label,
                                              average=average,
                                              warn_for=('specificity',),
                                              sample_weight=sample_weight)

    return s
