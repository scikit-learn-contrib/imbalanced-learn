.. _combine:

=======================================
Combination of over- and under-sampling
=======================================

.. currentmodule:: imblearn.over_sampling

We previously presented :class:`SMOTE` and showed that this method can generate
noisy samples by interpolating new points between marginal outliers and
inliers. This issue can be solved by cleaning the resulted space obtained
after over-sampling.

.. currentmodule:: imblearn.combine

In this regard, Tomek's link and edited nearest-neighbours are the two cleaning
methods which have been added pipeline after SMOTE over-sampling to obtain a
cleaner space. Therefore, imbalanced-learn implemented two ready-to-use class
which pipeline both over- and under-sampling methods: (i) :class:`SMOTETomek`
and (ii) :class:`SMOTEENN`.

These two classes can be used as any other sampler with identical parameters
than their former samplers::

  >>> from collections import Counter
  >>> from sklearn.datasets import make_classification
  >>> X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
  ...                            n_redundant=0, n_repeated=0, n_classes=3,
  ...                            n_clusters_per_class=1,
  ...                            weights=[0.01, 0.05, 0.94],
  ...                            class_sep=0.8, random_state=0)
  >>> print(Counter(y))
  Counter({2: 4674, 1: 262, 0: 64})
  >>> from imblearn.combine import SMOTEENN
  >>> smote_enn = SMOTEENN(random_state=0)
  >>> X_resampled, y_resampled = smote_enn.fit_sample(X, y)
  >>> print(Counter(y_resampled))
  Counter({1: 4381, 0: 4060, 2: 3502})
  >>> from imblearn.combine import SMOTETomek
  >>> smote_tomek = SMOTETomek(random_state=0)
  >>> X_resampled, y_resampled = smote_tomek.fit_sample(X, y)
  >>> print(Counter(y_resampled))
  Counter({1: 4566, 0: 4499, 2: 4413})

We can also see in the example below that :class:`SMOTEENN` tends to clean more
noisy samples than :class:`SMOTETomek`.

.. image:: ./auto_examples/combine/images/sphx_glr_plot_comparison_combine_001.png
   :target: ./auto_examples/combine/plot_comparison_combine.html
   :scale: 60
   :align: center

See :ref:`sphx_glr_auto_examples_combine_plot_smote_enn.py`,
:ref:`sphx_glr_auto_examples_combine_plot_smote_tomek.py`,
and
:ref:`sphx_glr_auto_examples_combine_plot_comparison_combine.py`.
