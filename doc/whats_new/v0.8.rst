.. _changes_0_8:

Version 0.8.1
=============

**September 29, 2020**

Changelog
---------

Maintenance
...........

- Make `imbalanced-learn` compatible with `scikit-learn` 1.0.
  :pr:`864` by :user:`Guillaume Lemaitre <glemaitre>`.

Version 0.8.0
=============

**February 18, 2021**

Changelog
---------

New features
............

- Add the the function
  :func:`imblearn.metrics.macro_averaged_mean_absolute_error` returning the
  average across class of the MAE. This metric is used in ordinal
  classification.
  :pr:`780` by :user:`Aurélien Massiot <AurelienMassiot>`.

- Add the class :class:`imblearn.metrics.pairwise.ValueDifferenceMetric` to
  compute pairwise distances between samples containing only categorical
  values.
  :pr:`796` by :user:`Guillaume Lemaitre <glemaitre>`.

- Add the class :class:`imblearn.over_sampling.SMOTEN` to over-sample data
  only containing categorical features.
  :pr:`802` by :user:`Guillaume Lemaitre <glemaitre>`.

- Add the possibility to pass any type of samplers in
  :class:`imblearn.ensemble.BalancedBaggingClassifier` unlocking the
  implementation of methods based on resampled bagging.
  :pr:`808` by :user:`Guillaume Lemaitre <glemaitre>`.

Enhancements
............

- Add option `output_dict` in
  :func:`imblearn.metrics.classification_report_imbalanced` to return a
  dictionary instead of a string.
  :pr:`770` by :user:`Guillaume Lemaitre <glemaitre>`.

- Added an option to generate smoothed bootstrap in
  :class:`imblearn.over_sampling.RandomOverSampler`. It is controls by the
  parameter `shrinkage`. This method is also known as Random Over-Sampling
  Examples (ROSE).
  :pr:`754` by :user:`Andrea Lorenzon <andrealorenzon>` and
  :user:`Guillaume Lemaitre <glemaitre>`.

Bug fixes
.........

- Fix a bug in :class:`imblearn.under_sampling.ClusterCentroids` where
  `voting="hard"` could have lead to select a sample from any class instead of
  the targeted class.
  :pr:`769` by :user:`Guillaume Lemaitre <glemaitre>`.

- Fix a bug in :class:`imblearn.FunctionSampler` where validation was performed
  even with `validate=False` when calling `fit`.
  :pr:`790` by :user:`Guillaume Lemaitre <glemaitre>`.

Maintenance
...........

- Remove requirements files in favour of adding the packages in the
  `extras_require` within the `setup.py` file.
  :pr:`816` by :user:`Guillaume Lemaitre <glemaitre>`.

- Change the website template to use `pydata-sphinx-theme`.
  :pr:`801` by :user:`Guillaume Lemaitre <glemaitre>`.

Deprecation
...........

- The context manager :func:`imblearn.utils.testing.warns` is deprecated in 0.8
  and will be removed 1.0.
  :pr:`815` by :user:`Guillaume Lemaitre <glemaitre>`.
