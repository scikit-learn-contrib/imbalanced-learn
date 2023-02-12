.. _changes_0_7:

Version 0.7.0
=============

**June 9, 2020**

Changelog
---------

Maintenance
...........

- Ensure that :class:`imblearn.pipeline.Pipeline` is working when `memory`
  is activated and `joblib==0.11`.
  :pr:`687` by :user:`Christos Aridas <chkoar>`.

- Refactor common test to use the dev tools from `scikit-learn` 0.23.
  :pr:`710` by :user:`Guillaume Lemaitre <glemaitre>`.

- Remove `FutureWarning` issued by `scikit-learn` 0.23.
  :pr:`710` by :user:`Guillaume Lemaitre <glemaitre>`.

- Impose keywords only argument as in `scikit-learn`.
  :pr:`721` by :user:`Guillaume Lemaitre <glemaitre>`.

Changed models
..............

The following models might give some different results due to changes:

- :class:`imblearn.ensemble.BalancedRandomForestClassifier`

Bug fixes
.........

- Change the default value `min_samples_leaf` to be consistent with
  scikit-learn.
  :pr:`711` by :user:`zerolfx <zerolfx>`.

- Fix a bug due to change in `scikit-learn` 0.23 in
  :class:`imblearn.metrics.make_index_balanced_accuracy`. The function was
  unusable.
  :pr:`710` by :user:`Guillaume Lemaitre <glemaitre>`.

- Raise a proper error message when only numerical or categorical features
  are given in :class:`imblearn.over_sampling.SMOTENC`.
  :pr:`720` by :user:`Guillaume Lemaitre <glemaitre>`.

- Fix a bug when the median of the standard deviation is null in
  :class:`imblearn.over_sampling.SMOTENC`.
  :pr:`675` by :user:`bganglia <bganglia>`.

Enhancements
............

- The classifier implemented in imbalanced-learn,
  :class:`imblearn.ensemble.BalancedBaggingClassifier`,
  :class:`imblearn.ensemble.BalancedRandomForestClassifier`,
  :class:`imblearn.ensemble.EasyEnsembleClassifier`, and
  :class:`imblearn.ensemble.RUSBoostClassifier`, accept `sampling_strategy`
  with the same key than in `y` without the need of encoding `y` in advance.
  :pr:`718` by :user:`Guillaume Lemaitre <glemaitre>`.

- Lazy import `keras` module when importing `imblearn.keras`
  :pr:`719` by :user:`Guillaume Lemaitre <glemaitre>`.

Deprecation
...........

- Deprecation of the parameters `n_jobs` in
  :class:`imblearn.under_sampling.ClusterCentroids` since it was used by
  :class:`sklearn.cluster.KMeans` which deprecated it.
  :pr:`710` by :user:`Guillaume Lemaitre <glemaitre>`.

- Deprecation of passing keyword argument by position similarly to
  `scikit-learn`.
  :pr:`721` by :user:`Guillaume lemaitre <glemaitre>`.
