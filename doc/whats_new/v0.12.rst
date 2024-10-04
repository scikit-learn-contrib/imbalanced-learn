.. _changes_0_12:

Version 0.12.4
==============

**October 4, 2024**

Changelog
---------

Compatibility
.............

- Compatibility with NumPy 2.0+
  :pr:`1097` by :user:`Guillaume Lemaitre <glemaitre>`.

Version 0.12.3
==============

**May 28, 2024**

Changelog
---------

Compatibility
.............

- Compatibility with scikit-learn 1.5
  :pr:`1074` and :pr:`1084` by :user:`Guillaume Lemaitre <glemaitre>`.

Version 0.12.2
==============

**March 31, 2024**

Changelog
---------

Bug fixes
.........

- Fix the way we check for a specific Python version in the test suite.
  :pr:`1075` by :user:`Guillaume Lemaitre <glemaitre>`.

Version 0.12.1
==============

**March 31, 2024**

Changelog
---------

Bug fixes
.........

- Fix a bug in :class:`~imblearn.under_sampling.InstanceHardnessThreshold` where
  `estimator` could not be a :class:`~sklearn.pipeline.Pipeline` object.
  :pr:`1049` by :user:`Gonenc Mogol <gmogol>`.

Compatibility
.............

- Do not use `distutils` in tests due to deprecation.
  :pr:`1065` by :user:`Michael R. Crusoe <mr-c>`.

- Fix the scikit-learn import in tests to be compatible with version 1.4.1.post1.
  :pr:`1073` by :user:`Guillaume Lemaitre <glemaitre>`.

- Fix test to be compatible with Python 3.13.
  :pr:`1073` by :user:`Guillaume Lemaitre <glemaitre>`.

Version 0.12.0
==============

**January 24, 2024**

Changelog
---------

Bug fixes
.........

- Fix a bug in :class:`~imblearn.over_sampling.SMOTENC` where the entries of the
  one-hot encoding should be divided by `sqrt(2)` and not `2`, taking into account that
  they are plugged into an Euclidean distance computation.
  :pr:`1014` by :user:`Guillaume Lemaitre <glemaitre>`.

- Raise an informative error message when all support vectors are tagged as noise in
  :class:`~imblearn.over_sampling.SVMSMOTE`.
  :pr:`1016` by :user:`Guillaume Lemaitre <glemaitre>`.

- Fix a bug in :class:`~imblearn.over_sampling.SMOTENC` where the median of standard
  deviation of the continuous features was only computed on the minority class. Now,
  we are computing this statistic for each class that is up-sampled.
  :pr:`1015` by :user:`Guillaume Lemaitre <glemaitre>`.

- Fix a bug in :class:`~imblearn.over_sampling.SMOTENC` such that the case where
  the median of standard deviation of the continuous features is null is handled
  in the multiclass case as well.
  :pr:`1015` by :user:`Guillaume Lemaitre <glemaitre>`.

- Fix a bug in :class:`~imblearn.over_sampling.BorderlineSMOTE` version 2 where samples
  should be generated from the whole dataset and not only from the minority class.
  :pr:`1023` by :user:`Guillaume Lemaitre <glemaitre>`.

- Fix a bug in :class:`~imblearn.under_sampling.NeighbourhoodCleaningRule` where the
  `kind_sel="all"` was not working as explained in the literature.
  :pr:`1012` by :user:`Guillaume Lemaitre <glemaitre>`.

- Fix a bug in :class:`~imblearn.under_sampling.NeighbourhoodCleaningRule` where the
  `threshold_cleaning` ratio was multiplied on the total number of samples instead of
  the number of samples in the minority class.
  :pr:`1012` by :user:`Guillaume Lemaitre <glemaitre>`.

- Fix a bug in :class:`~imblearn.under_sampling.RandomUnderSampler` and
  :class:`~imblearn.over_sampling.RandomOverSampler` where a column containing only
  NaT was not handled correctly.
  :pr:`1059` by :user:`Guillaume Lemaitre <glemaitre>`.

Compatibility
.............

- :class:`~imblearn.ensemble.BalancedRandomForestClassifier` now support missing values
  and monotonic constraints if scikit-learn >= 1.4 is installed.

- :class:`~imblearn.pipeline.Pipeline` support metadata routing if scikit-learn >= 1.4
  is installed.

- Compatibility with scikit-learn 1.4.
  :pr:`1058` by :user:`Guillaume Lemaitre <glemaitre>`.

Deprecations
............

- Deprecate `estimator_` argument in favor of `estimators_` for the classes
  :class:`~imblearn.under_sampling.CondensedNearestNeighbour` and
  :class:`~imblearn.under_sampling.OneSidedSelection`. `estimator_` will be removed
  in 0.14.
  :pr:`1011` by :user:`Guillaume Lemaitre <glemaitre>`.

- Deprecate `kind_sel` in :class:`~imblearn.under_sampling.NeighbourhoodCleaningRule.
  It will be removed in 0.14. The parameter does not have any effect.
  :pr:`1012` by :user:`Guillaume Lemaitre <glemaitre>`.

Enhancements
............

- Allows to output dataframe with sparse format if provided as input.
  :pr:`1059` by :user:`ts2095 <ts2095>`.
