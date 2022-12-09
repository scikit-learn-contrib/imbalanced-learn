.. _changes_0_6_2:

Version 0.6.2
==============

**February 16, 2020**

This is a bug-fix release to resolve some issues regarding the handling the
input and the output format of the arrays.

Changelog
---------

- Allow column vectors to be passed as targets.
  :pr:`673` by :user:`Christos Aridas <chkoar>`.

- Better input/output handling for pandas, numpy and plain lists.
  :pr:`681` by :user:`Christos Aridas <chkoar>`.

.. _changes_0_6_1:

Version 0.6.1
==============

**December 7, 2019**

This is a bug-fix release to primarily resolve some packaging issues in version
0.6.0. It also includes minor documentation improvements and some bug fixes.

Changelog
---------

Bug fixes
.........

- Fix a bug in :class:`imblearn.ensemble.BalancedRandomForestClassifier`
  leading to a wrong number of samples used during fitting due `max_samples`
  and therefore a bad computation of the OOB score.
  :pr:`656` by :user:`Guillaume Lemaitre <glemaitre>`.

.. _changes_0_6:

Version 0.6.0
=============

**December 5, 2019**

Changelog
---------

Changed models
..............

The following models might give some different sampling due to changes in
scikit-learn:

- :class:`imblearn.under_sampling.ClusterCentroids`
- :class:`imblearn.under_sampling.InstanceHardnessThreshold`

The following samplers will give different results due to change linked to
the random state internal usage:

- :class:`imblearn.over_sampling.ADASYN`
- :class:`imblearn.over_sampling.SMOTENC`

Bug fixes
.........

- :class:`imblearn.under_sampling.InstanceHardnessThreshold` now take into
  account the `random_state` and will give deterministic results. In addition,
  `cross_val_predict` is used to take advantage of the parallelism.
  :pr:`599` by :user:`Shihab Shahriar Khan <Shihab-Shahriar>`.

- Fix a bug in :class:`imblearn.ensemble.BalancedRandomForestClassifier`
  leading to a wrong computation of the OOB score.
  :pr:`656` by :user:`Guillaume Lemaitre <glemaitre>`.

Maintenance
...........

- Update imports from scikit-learn after that some modules have been privatize.
  The following import have been changed:
  :class:`sklearn.ensemble._base._set_random_states`,
  :class:`sklearn.ensemble._forest._parallel_build_trees`,
  :class:`sklearn.metrics._classification._check_targets`,
  :class:`sklearn.metrics._classification._prf_divide`,
  :class:`sklearn.utils.Bunch`,
  :class:`sklearn.utils._safe_indexing`,
  :class:`sklearn.utils._testing.assert_allclose`,
  :class:`sklearn.utils._testing.assert_array_equal`,
  :class:`sklearn.utils._testing.SkipTest`.
  :pr:`617` by :user:`Guillaume Lemaitre <glemaitre>`.

- Synchronize :mod:`imblearn.pipeline` with :mod:`sklearn.pipeline`.
  :pr:`620` by :user:`Guillaume Lemaitre <glemaitre>`.

- Synchronize :class:`imblearn.ensemble.BalancedRandomForestClassifier` and add
  parameters `max_samples` and `ccp_alpha`.
  :pr:`621` by :user:`Guillaume Lemaitre <glemaitre>`.

Enhancement
...........

- :class:`imblearn.under_sampling.RandomUnderSampling`,
  :class:`imblearn.over_sampling.RandomOverSampling`,
  :class:`imblearn.datasets.make_imbalance` accepts Pandas DataFrame in and
  will output Pandas DataFrame. Similarly, it will accepts Pandas Series in and
  will output Pandas Series.
  :pr:`636` by :user:`Guillaume Lemaitre <glemaitre>`.

- :class:`imblearn.FunctionSampler` accepts a parameter ``validate`` allowing
  to check or not the input ``X`` and ``y``.
  :pr:`637` by :user:`Guillaume Lemaitre <glemaitre>`.

- :class:`imblearn.under_sampling.RandomUnderSampler`,
  :class:`imblearn.over_sampling.RandomOverSampler` can resample when non
  finite values are present in ``X``.
  :pr:`643` by :user:`Guillaume Lemaitre <glemaitre>`.

- All samplers will output a Pandas DataFrame if a Pandas DataFrame was given
  as an input.
  :pr:`644` by :user:`Guillaume Lemaitre <glemaitre>`.

- The samples generation in
  :class:`imblearn.over_sampling.ADASYN`,
  :class:`imblearn.over_sampling.SMOTE`,
  :class:`imblearn.over_sampling.BorderlineSMOTE`,
  :class:`imblearn.over_sampling.SVMSMOTE`,
  :class:`imblearn.over_sampling.KMeansSMOTE`,
  :class:`imblearn.over_sampling.SMOTENC` is now vectorize with giving
  an additional speed-up when `X` in sparse.
  :pr:`596` and :pr:`649` by :user:`Matt Eding <MattEding>`.

Deprecation
...........

- The following classes have been removed after 2 deprecation cycles:
  `ensemble.BalanceCascade` and `ensemble.EasyEnsemble`.
  :pr:`617` by :user:`Guillaume Lemaitre <glemaitre>`.

- The following functions have been removed after 2 deprecation cycles:
  `utils.check_ratio`.
  :pr:`617` by :user:`Guillaume Lemaitre <glemaitre>`.

- The parameter `ratio` and `return_indices` has been removed from all
  samplers.
  :pr:`617` by :user:`Guillaume Lemaitre <glemaitre>`.

- The parameters `m_neighbors`, `out_step`, `kind`, `svm_estimator`
  have been removed from the :class:`imblearn.over_sampling.SMOTE`.
  :pr:`617` by :user:`Guillaume Lemaitre <glemaitre>`.
