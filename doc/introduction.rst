.. _introduction:

============
Introduction
============

.. _api_imblearn:

API's of imbalanced-learn samplers
----------------------------------

The available samplers follows the scikit-learn API using the base estimator
and adding a sampling functionality through the ``sample`` method:

:Estimator:

    The base object, implements a ``fit`` method to learn from data, either::

      estimator = obj.fit(data, targets)

:Resampler:

    To resample a data sets, each sampler implements::

      data_resampled, targets_resampled = obj.fit_resample(data, targets)

Imbalanced-learn samplers accept the same inputs that in scikit-learn:

* ``data``: array-like (2-D list, pandas.Dataframe, numpy.array) or sparse
  matrices;
* ``targets``: array-like (1-D list, pandas.Series, numpy.array).

.. topic:: Sparse input

   For sparse input the data is **converted to the Compressed Sparse Rows
   representation** (see ``scipy.sparse.csr_matrix``) before being fed to the
   sampler. To avoid unnecessary memory copies, it is recommended to choose the
   CSR representation upstream.

.. _problem_statement:

Problem statement regarding imbalanced data sets
------------------------------------------------

The learning phase and the subsequent prediction of machine learning algorithms
can be affected by the problem of imbalanced data set. The balancing issue
corresponds to the difference of the number of samples in the different
classes. We illustrate the effect of training a linear SVM classifier with
different level of class balancing.

.. image:: ./auto_examples/over-sampling/images/sphx_glr_plot_comparison_over_sampling_001.png
   :target: ./auto_examples/over-sampling/plot_comparison_over_sampling.html
   :scale: 60
   :align: center

As expected, the decision function of the linear SVM is highly impacted. With a
greater imbalanced ratio, the decision function favor the class with the larger
number of samples, usually referred as the majority class.
