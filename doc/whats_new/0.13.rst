.. _changes_0_13:

Version 0.13.0
==============

**December 20, 2024**

Changelog
---------

Bug fixes
.........

- Fix `get_metadata_routing` in :class:`~imblearn.pipeline.Pipeline` such that one
  can use a sampler with metadata routing.
  :pr:`1115` by :user:`Guillaume Lemaitre <glemaitre>`.

Compatibility
.............

- Compatibility with scikit-learn 1.6
  :pr:`1109` by :user:`Guillaume Lemaitre <glemaitre>`.

Deprecations
............

- :class:`~imblearn.pipeline.Pipeline` now uses
  :func:`~sklearn.utils.check_is_fitted` instead of
  :func:`~sklearn.utils.check_fitted` to check if the pipeline is fitted. In 0.15, it
  will raise an error instead of a warning.
  :pr:`1109` by :user:`Guillaume Lemaitre <glemaitre>`.

- `algorithm` parameter in :class:`~imblearn.ensemble.RUSBoostClassifier` is now
  deprecated and will be removed in 0.14.
  :pr:`1109` by :user:`Guillaume Lemaitre <glemaitre>`.
