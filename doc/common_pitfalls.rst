.. _common_pitfalls:

=========================================
Common pitfalls and recommended practices
=========================================

This section is a complement to the documentation given
`[here] <https://scikit-learn.org/dev/common_pitfalls.html>`_ in scikit-learn.
Indeed, we will highlight the issue of misusing resampling, leading to a
**data leakage**. Due to this leakage, the performance of a model reported
will be over-optimistic.

Data leakage
============

As mentioned in the scikit-learn documentation, Data leakage occurs when
information that would not be available at prediction time is used when
building the model.

In a resampling setting, the common pitfall is to resample the **entire**
dataset before to split the data into a train-test split. Doing such processing
leads to two issues:

* the model will not be tested on a dataset with class distribution similar
  to the real use-case. Indeed, by resampling the entire dataset, both the
  training and testing set will be potentially balanced while the model should
  be tested on the natural imbalanced dataset to evaluate the potential bias
  of the model;
* the resampling procedure might use information about samples in the dataset
  to either generate or select some of the samples. Therefore, we might use
  information of samples which will be later used as testing samples which
  is the typical data leakage issue.

We will demonstrate the wrong and right ways to do some sampling and emphasize
the tools that one should use, avoiding to fall in the trap.

::

    >>> import numpy as np
    >>> from sklearn.datasets import load_breast_cancer
    >>> X, y = load_breast_cancer(return_X_y=True)
    >>> idx_positive = np.flatnonzero(y == 1)
    >>> idx_negative = np.flatnonzero(y == 0)
    >>> idx_selected = np.hstack([idx_negative, idx_positive[:25]])
    >>> X, y = X[idx_selected], y[idx_selected]
    >>> # only use 2 features to make the problem even harder
    >>> X = X[:, :2]

**Wrong**

::
