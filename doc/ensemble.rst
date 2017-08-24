.. _ensemble:

====================
Ensemble of samplers
====================

.. currentmodule:: imblearn.ensemble

An imbalanced data set can be balanced by creating several balanced
subsets. The module :mod:`imblearn.ensemble` allows to create such sets.

:class:`EasyEnsemble` creates an ensemble of data set by randomly
under-sampling the original set::

  >>> from collections import Counter
  >>> from sklearn.datasets import make_classification
  >>> X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
  ...                            n_redundant=0, n_repeated=0, n_classes=3,
  ...                            n_clusters_per_class=1,
  ...                            weights=[0.01, 0.05, 0.94],
  ...                            class_sep=0.8, random_state=0)
  >>> print(sorted(Counter(y).items()))
  [(0, 64), (1, 262), (2, 4674)]
  >>> from imblearn.ensemble import EasyEnsemble
  >>> ee = EasyEnsemble(random_state=0, n_subsets=10)
  >>> X_resampled, y_resampled = ee.fit_sample(X, y)
  >>> print(X_resampled.shape)
  (10, 192, 2)
  >>> print(sorted(Counter(y_resampled[0]).items()))
  [(0, 64), (1, 64), (2, 64)]

:class:`EasyEnsemble` has two important parameters: (i) ``n_subsets`` will be
used to return number of subset and (ii) ``replacement`` to randomly sample
with or without replacement.

:class:`BalanceCascade` differs from the previous method by using a classifier
(using the parameter ``estimator``) to ensure that misclassified samples can
again be selected for the next subset. In fact, the classifier play the role of
a "smart" replacement method. The maximum number of subset can be set using the
parameter ``n_max_subset`` and an additional bootstraping can be activated with
``bootstrap`` set to ``True``::

  >>> from imblearn.ensemble import BalanceCascade
  >>> from sklearn.linear_model import LogisticRegression
  >>> bc = BalanceCascade(random_state=0,
  ...                     estimator=LogisticRegression(random_state=0),
  ...                     n_max_subset=4)
  >>> X_resampled, y_resampled = bc.fit_sample(X, y)
  >>> print(X_resampled.shape)
  (4, 192, 2)
  >>> print(sorted(Counter(y_resampled[0]).items()))
  [(0, 64), (1, 64), (2, 64)]

See
:ref:`sphx_glr_auto_examples_ensemble_plot_easy_ensemble.py` and
:ref:`sphx_glr_auto_examples_ensemble_plot_balance_cascade.py`.
