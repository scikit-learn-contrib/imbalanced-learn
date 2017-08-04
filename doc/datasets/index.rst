.. _datasets:

=========================
Dataset loading utilities
=========================

.. currentmodule:: imblearn.datasets

The ``imblearn.datasets`` package is complementing the the
``sklearn.datasets`` package. The package provide both: (i) a set of
imbalanced datasets to perform systematic benchmark and (ii) a utility
to create an imbalanced dataset from an original balanced dataset.

Imbalanced datasets for benchmark
=================================

Imbalanced generator
====================

:func:`make_imbalance` turns an original dataset into an imbalanced
dataset. This behaviour is driven by the parameter ``ratio`` which behave
similarly to other resampling algorithm. ``ratio`` can be given as a dictionary
where the key corresponds to the class and the value is the the number of
samples in the class::

  >>> from collections import Counter
  >>> from sklearn.datasets import load_iris
  >>> from imblearn.datasets import make_imbalance
  >>> iris = load_iris()
  >>> ratio = {0: 20, 1: 30, 2: 40}
  >>> X_imb, y_imb = make_imbalance(iris.data, iris.target, ratio=ratio)
  >>> Counter(y_imb)
  Counter({2: 40, 1: 30, 0: 20})

Note that all samples of a class is pass-through if the class is not mentioned
in the dictionary::

  >>> ratio = {0: 10}
  >>> X_imb, y_imb = make_imbalance(iris.data, iris.target, ratio=ratio)
  >>> Counter(y_imb)
  Counter({1: 50, 2: 50, 0: 10})

Instead of a dictionary, a function can be defined and directly pass to
``ratio``::

  >>> def ratio_multiplier(y):
  ...     multiplier = {0: 0.5, 1: 0.7, 2: 0.95}
  ...     target_stats = Counter(y)
  ...     for key, value in target_stats.items():
  ...         target_stats[key] = int(value * multiplier[key])
  ...     return target_stats
  >>> X_imb, y_imb = make_imbalance(iris.data, iris.target,
  ...                               ratio=ratio_multiplier)
  >>> Counter(y_imb)
  Counter({2: 47, 1: 35, 0: 25})


See :ref:`sphx_glr_auto_examples_datasets_plot_make_imbalance.py` and
:ref:`sphx_glr_auto_examples_plot_ratio_usage.py`.
