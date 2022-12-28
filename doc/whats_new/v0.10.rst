.. _changes_0_10:

Version 0.10.1
==============

**December 28, 2022**

Changelog
---------

Bug fixes
.........

- Fix a regression in over-sampler where the string `minority` was rejected as
  an unvalid sampling strategy.
  :pr:`964` by :user:`Prakhyath Bhandary <Prakhyath07>`.

Version 0.10.0
==============

**December 9, 2022**

Changelog
---------

Bug fixes
.........

- Make sure that :class:`~imblearn.utils._docstring.Substitution` is
  working with `python -OO` that replace `__doc__` by `None`.
  :pr:`953` bu :user:`Guillaume Lemaitre <glemaitre>`.

Compatibility
.............

- Maintenance release for be compatible with scikit-learn >= 1.0.2.
  :pr:`946`, :pr:`947`, :pr:`949` by :user:`Guillaume Lemaitre <glemaitre>`.

- Add support for automatic parameters validation as in scikit-learn >= 1.2.
  :pr:`955` by :user:`Guillaume Lemaitre <glemaitre>`.

- Add support for `feature_names_in_` as well as `get_feature_names_out` for
  all samplers.
  :pr:`959` by :user:`Guillaume Lemaitre <glemaitre>`.

Deprecation
...........

- The parameter `n_jobs` has been deprecated from the classes
  :class:`~imblearn.over_sampling.ADASYN`,
  :class:`~imblearn.over_sampling.BorderlineSMOTE`,
  :class:`~imblearn.over_sampling.SMOTE`,
  :class:`~imblearn.over_sampling.SMOTENC`,
  :class:`~imblearn.over_sampling.SMOTEN`, and
  :class:`~imblearn.over_sampling.SVMSMOTE`. Instead, pass a nearest neighbors
  estimator where `n_jobs` is set.
  :pr:`887` by :user:`Guillaume Lemaitre <glemaitre>`.

- The parameter `base_estimator` is deprecated and will be removed in version
  0.12. It is impacted the following classes:
  :class:`~imblearn.ensemble.BalancedBaggingClassifier`,
  :class:`~imblearn.ensemble.EasyEnsembleClassifier`,
  :class:`~imblearn.ensemble.RUSBoostClassifier`.
  :pr:`946` by :user:`Guillaume Lemaitre <glemaitre>`.


Enhancements
............

- Add support to accept compatible `NearestNeighbors` objects by only
  duck-typing. For instance, it allows to accept cuML instances.
  :pr:`858` by :user:`NV-jpt <NV-jpt>` and
  :user:`Guillaume Lemaitre <glemaitre>`.
