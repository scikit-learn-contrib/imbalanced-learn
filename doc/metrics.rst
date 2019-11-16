.. _metrics:

=======
Metrics
=======

.. currentmodule:: imblearn.metrics

Currently, scikit-learn only offers the
``sklearn.metrics.balanced_accuracy_score`` (in 0.20) as metric to deal with
imbalanced datasets. The module :mod:`imblearn.metrics` offers a couple of
other metrics which are used in the literature to evaluate the quality of
classifiers.

.. _sensitivity_specificity:

Sensitivity and specificity metrics
-----------------------------------

Sensitivity and specificity are metrics which are well known in medical
imaging. Sensitivity (also called true positive rate or recall) is the
proportion of the positive samples which is well classified while specificity
(also called true negative rate) is the proportion of the negative samples
which are well classified. Therefore, depending of the field of application,
either the sensitivity/specificity or the precision/recall pair of metrics are
used.

Currently, only the `precision and recall metrics
<http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html>`_
are implemented in scikit-learn. :func:`sensitivity_specificity_support`,
:func:`sensitivity_score`, and :func:`specificity_score` add the possibility to
use those metrics.

.. _imbalanced_metrics:

Additional metrics specific to imbalanced datasets
--------------------------------------------------

The :func:`geometric_mean_score`
:cite:`barandela2003strategies,kubat1997addressing` is the root of the product
of class-wise sensitivity. This measure tries to maximize the accuracy on each
of the classes while keeping these accuracies balanced.

The :func:`make_index_balanced_accuracy` :cite:`garcia2012effectiveness` can
wrap any metric and give more importance to a specific class using the
parameter ``alpha``.
