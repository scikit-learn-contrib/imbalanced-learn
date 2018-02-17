.. _miscellaneous:

======================
Miscellaneous samplers
======================

.. currentmodule:: imblearn

.. _function_sampler:

Custom samplers
---------------

A fully customized sampler, :class:`FunctionSampler`, is available in
imbalanced-learn such that you can fast prototype your own sampler by defining
a single function. Additional parameters can be added using the attribute
``kw_args`` which accepts a dictionary. The following example illustrates how
to retain the 10 first elements of the array ``X`` and ``y``::

  >>> import numpy as np
  >>> from imblearn import FunctionSampler
  >>> from sklearn.datasets import make_classification
  >>> X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
  ...                            n_redundant=0, n_repeated=0, n_classes=3,
  ...                            n_clusters_per_class=1,
  ...                            weights=[0.01, 0.05, 0.94],
  ...                            class_sep=0.8, random_state=0)
  >>> def func(X, y):
  ...   return X[:10], y[:10]
  >>> sampler = FunctionSampler(func=func)
  >>> X_res, y_res = sampler.fit_sample(X, y)
  >>> np.all(X_res == X[:10])
  True
  >>> np.all(y_res == y[:10])
  True

We illustrate the use of such sampler to implement an outlier rejection
estimator which can be easily used within a
:class:`imblearn.pipeline.Pipeline`:
:ref:`sphx_glr_auto_examples_plot_outlier_rejections.py`
