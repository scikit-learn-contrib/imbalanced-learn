.. _changes_0_4:

Version 0.4.2
=============

**October 21, 2018**

Changelog
---------

Bug fixes
.........

- Fix a bug in :class:`imblearn.over_sampling.SMOTENC` in which the the median
  of the standard deviation instead of half of the median of the standard
  deviation.
  By :user:`Guillaume Lemaitre <glemaitre>` in :issue:`491`.

- Raise an error when passing target  which is not supported, i.e. regression
  target or multilabel targets. Imbalanced-learn does not support this case.
  By :user:`Guillaume Lemaitre <glemaitre>` in :issue:`490`.

- Fix a bug in :class:`imblearn.over_sampling.SMOTENC` in which a sparse
  matrices were densify during ``inverse_transform``.
  By :user:`Guillaume Lemaitre <glemaitre>` in :issue:`495`.

- Fix a bug in :class:`imblearn.over_sampling.SMOTE_NC` in which a the tie
  breaking was wrongly sampling.
  By :user:`Guillaume Lemaitre <glemaitre>` in :issue:`497`.

Version 0.4
===========

**October 12, 2018**

.. warning::

    Version 0.4 is the last version of imbalanced-learn to support Python 2.7
    and Python 3.4. Imbalanced-learn 0.5 will require Python 3.5 or higher.

Highlights
----------

This release brings its set of new feature as well as some API changes to
strengthen the foundation of imbalanced-learn.

As new feature, 2 new modules :mod:`imblearn.keras` and
:mod:`imblearn.tensorflow` have been added in which imbalanced-learn samplers
can be used to generate balanced mini-batches.

The module :mod:`imblearn.ensemble` has been consolidated with new classifier:
:class:`imblearn.ensemble.BalancedRandomForestClassifier`,
:class:`imblearn.ensemble.EasyEnsembleClassifier`,
:class:`imblearn.ensemble.RUSBoostClassifier`.

Support for string has been added in
:class:`imblearn.over_sampling.RandomOverSampler` and
:class:`imblearn.under_sampling.RandomUnderSampler`. In addition, a new class
:class:`imblearn.over_sampling.SMOTENC` allows to generate sample with data
sets containing both continuous and categorical features.

The :class:`imblearn.over_sampling.SMOTE` has been simplified and break down
to 2 additional classes:
:class:`imblearn.over_sampling.SVMSMOTE` and
:class:`imblearn.over_sampling.BorderlineSMOTE`.

There is also some changes regarding the API:
the parameter ``sampling_strategy`` has been introduced to replace the
``ratio`` parameter. In addition, the ``return_indices`` argument has been
deprecated and all samplers will exposed a ``sample_indices_`` whenever this is
possible.

Changelog
---------

API
...

- Replace the parameter ``ratio`` by ``sampling_strategy``. :issue:`411` by
  :user:`Guillaume Lemaitre <glemaitre>`.

- Enable to use a ``float`` with binary classification for
  ``sampling_strategy``. :issue:`411` by :user:`Guillaume Lemaitre <glemaitre>`.

- Enable to use a ``list`` for the cleaning methods to specify the class to
  sample. :issue:`411` by :user:`Guillaume Lemaitre <glemaitre>`.

- Replace ``fit_sample`` by ``fit_resample``. An alias is still available for
  backward compatibility. In addition, ``sample`` has been removed to avoid
  resampling on different set of data.
  :issue:`462` by :user:`Guillaume Lemaitre <glemaitre>`.

New features
............

- Add a :mod:`keras` and :mod:`tensorflow` modules to create balanced
  mini-batches generator.
  :issue:`409` by :user:`Guillaume Lemaitre <glemaitre>`.

- Add :class:`imblearn.ensemble.EasyEnsembleClassifier` which create a bag of
  AdaBoost classifier trained on balanced bootstrap samples.
  :issue:`455` by :user:`Guillaume Lemaitre <glemaitre>`.

- Add :class:`imblearn.ensemble.BalancedRandomForestClassifier` which balanced
  each bootstrap provided to each tree of the forest.
  :issue:`459` by :user:`Guillaume Lemaitre <glemaitre>`.

- Add :class:`imblearn.ensemble.RUSBoostClassifier` which applied a random
  under-sampling stage before each boosting iteration of AdaBoost.
  :issue:`469` by :user:`Guillaume Lemaitre <glemaitre>`.

- Add :class:`imblern.over_sampling.SMOTENC` which generate synthetic samples
  on data set with heterogeneous data type (continuous and categorical
  features).
  :issue:`412` by :user:`Denis Dudnik <ddudnik>` and
  :user:`Guillaume Lemaitre <glemaitre>`.

Enhancement
...........

- Add a documentation node to create a balanced random forest from a balanced
  bagging classifier. :issue:`372` by :user:`Guillaume Lemaitre <glemaitre>`.

- Document the metrics to evaluate models on imbalanced dataset. :issue:`367`
  by :user:`Guillaume Lemaitre <glemaitre>`.

- Add support for one-vs-all encoded target to support keras. :issue:`409` by
  :user:`Guillaume Lemaitre <glemaitre>`.

- Adding specific class for borderline and SVM SMOTE using
  :class:`BorderlineSMOTE` and :class:`SVMSMOTE`.
  :issue:`440` by :user:`Guillaume Lemaitre <glemaitre>`.

- Allow :class:`imblearn.over_sampling.RandomOverSampler` can return indices
  using the attributes ``return_indices``.
  :issue:`439` by :user:`Hugo Gascon<hgascon>` and
  :user:`Guillaume Lemaitre <glemaitre>`.

- Allow :class:`imblearn.under_sampling.RandomUnderSampler` and
  :class:`imblearn.over_sampling.RandomOverSampler` to sample object array
  containing strings.
  :issue:`451` by :user:`Guillaume Lemaitre <glemaitre>`.

Bug fixes
.........

- Fix bug in :func:`metrics.classification_report_imbalanced` for which
  `y_pred` and `y_true` where inversed. :issue:`394` by :user:`Ole Silvig
  <klizter>.`

- Fix bug in ADASYN to consider only samples from the current class when
  generating new samples. :issue:`354` by :user:`Guillaume Lemaitre
  <glemaitre>`.

- Fix bug which allow for sorted behavior of ``sampling_strategy`` dictionary
  and thus to obtain a deterministic results when using the same random state.
  :issue:`447` by :user:`Guillaume Lemaitre <glemaitre>`.

- Force to clone scikit-learn estimator passed as attributes to samplers.
  :issue:`446` by :user:`Guillaume Lemaitre <glemaitre>`.

- Fix bug which was not preserving the dtype of X and y when generating
  samples.
  :issue:`450` by :user:`Guillaume Lemaitre <glemaitre>`.

- Add the option to pass a ``Memory`` object to :func:`make_pipeline` like
  in :class:`pipeline.Pipeline` class.
  :issue:`458` by :user:`Christos Aridas <chkoar>`.

Maintenance
...........

- Remove deprecated parameters in 0.2 - :issue:`331` by :user:`Guillaume
  Lemaitre <glemaitre>`.

- Make some modules private.
  :issue:`452` by :user:`Guillaume Lemaitre <glemaitre>`.

- Upgrade requirements to scikit-learn 0.20.
  :issue:`379` by :user:`Guillaume Lemaitre <glemaitre>`.

- Catch deprecation warning in testing.
  :issue:`441` by :user:`Guillaume Lemaitre <glemaitre>`.

- Refactor and impose `pytest` style tests.
  :issue:`470` by :user:`Guillaume Lemaitre <glemaitre>`.

Documentation
.............

- Remove some docstring which are not necessary.
  :issue:`454` by :user:`Guillaume Lemaitre <glemaitre>`.

- Fix the documentation of the ``sampling_strategy`` parameters when used as a
  float.
  :issue:`480` by :user:`Guillaume Lemaitre <glemaitre>`.

Deprecation
...........

- Deprecate ``ratio`` in favor of ``sampling_strategy``. :issue:`411` by
  :user:`Guillaume Lemaitre <glemaitre>`.

- Deprecate the use of a ``dict`` for cleaning methods. a ``list`` should be
  used. :issue:`411` by :user:`Guillaume Lemaitre <glemaitre>`.

- Deprecate ``random_state`` in :class:`imblearn.under_sampling.NearMiss`,
  :class:`imblearn.under_sampling.EditedNearestNeighbors`,
  :class:`imblearn.under_sampling.RepeatedEditedNearestNeighbors`,
  :class:`imblearn.under_sampling.AllKNN`,
  :class:`imblearn.under_sampling.NeighbourhoodCleaningRule`,
  :class:`imblearn.under_sampling.InstanceHardnessThreshold`,
  :class:`imblearn.under_sampling.CondensedNearestNeighbours`.

- Deprecate ``kind``, ``out_step``, ``svm_estimator``, ``m_neighbors`` in
  :class:`imblearn.over_sampling.SMOTE`. User should use
  :class:`imblearn.over_sampling.SVMSMOTE` and
  :class:`imblearn.over_sampling.BorderlineSMOTE`.
  :issue:`440` by :user:`Guillaume Lemaitre <glemaitre>`.

- Deprecate :class:`imblearn.ensemble.EasyEnsemble` in favor of meta-estimator
  :class:`imblearn.ensemble.EasyEnsembleClassifier` which follow the exact
  algorithm described in the literature.
  :issue:`455` by :user:`Guillaume Lemaitre <glemaitre>`.

- Deprecate :class:`imblearn.ensemble.BalanceCascade`.
  :issue:`472` by :user:`Guillaume Lemaitre <glemaitre>`.

- Deprecate ``return_indices`` in all samplers. Instead, an attribute
  ``sample_indices_`` is created whenever the sampler is selecting a subset of
  the original samples.
  :issue:`474` by :user:`Guillaume Lemaitre <glemaitre`.
