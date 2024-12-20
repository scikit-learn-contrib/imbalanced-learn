.. _introduction:

============
Introduction
============

.. _api_imblearn:

API's of imbalanced-learn samplers
----------------------------------

The available samplers follow the
`scikit-learn API <https://scikit-learn.org/stable/getting_started.html#fitting-and-predicting-estimator-basics>`_
using the base estimator
and incorporating a sampling functionality via the ``sample`` method:

:Estimator:

    The base object, implements a ``fit`` method to learn from data::

      estimator = obj.fit(data, targets)

:Resampler:

    To resample a data sets, each sampler implements a ``fit_resample`` method::

      data_resampled, targets_resampled = obj.fit_resample(data, targets)

Imbalanced-learn samplers accept the same inputs as scikit-learn estimators:

* `data`, 2-dimensional array-like structures, such as:
   * Python's list of lists :class:`list`,
   * Numpy arrays :class:`numpy.ndarray`,
   * Panda dataframes :class:`pandas.DataFrame`,
   * Scipy sparse matrices :class:`scipy.sparse.csr_matrix` or :class:`scipy.sparse.csc_matrix`;

* `targets`, 1-dimensional array-like structures, such as:
   * Numpy arrays :class:`numpy.ndarray`,
   * Pandas series :class:`pandas.Series`.

The output will be of the following type:

* `data_resampled`, 2-dimensional aray-like structures, such as:
   * Numpy arrays :class:`numpy.ndarray`,
   * Pandas dataframes :class:`pandas.DataFrame`,
   * Scipy sparse matrices :class:`scipy.sparse.csr_matrix` or :class:`scipy.sparse.csc_matrix`;

* `targets_resampled`, 1-dimensional array-like structures, such as:
   * Numpy arrays :class:`numpy.ndarray`,
   * Pandas series :class:`pandas.Series`.

.. topic:: Pandas in/out

   Unlike scikit-learn, imbalanced-learn provides support for pandas in/out.
   Therefore providing a dataframe, will output as well a dataframe.

.. topic:: Sparse input

   For sparse input the data is **converted to the Compressed Sparse Rows
   representation** (see ``scipy.sparse.csr_matrix``) before being fed to the
   sampler. To avoid unnecessary memory copies, it is recommended to choose the
   CSR representation upstream.

.. _problem_statement:

Problem statement regarding imbalanced data sets
------------------------------------------------

The learning and prediction phrases of machine learning algorithms
can be impacted by the issue of **imbalanced datasets**. This imbalance
refers to the difference in the number of samples across different classes.
We demonstrate the effect of training a `Logistic Regression classifier
<https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html>`_
with varying levels of class balancing by adjusting their weights.

.. image:: ./auto_examples/over-sampling/images/sphx_glr_plot_comparison_over_sampling_001.png
   :target: ./auto_examples/over-sampling/plot_comparison_over_sampling.html
   :scale: 60
   :align: center

As expected, the decision function of the Logistic Regression classifier varies significantly
depending on how imbalanced the data is. With a greater imbalance ratio, the decision function
tends to favour the class with the larger number of samples, usually referred to as the
**majority class**.
