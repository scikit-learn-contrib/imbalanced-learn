.. _changes_0_11:

Version 0.11.0
==============

**July 8, 2023**

Changelog
---------

Bug fixes
.........

- Fix a bug in :func:`~imblearn.metrics.classification_report_imbalanced` where the
  parameter `target_names` was not taken into account when `output_dict=True`.
  :pr:`989` by :user:`AYY7 <AYY7>`.

- :class:`~imblearn.over_sampling.SMOTENC` now handles mix types of data type such as
  `bool` and `pd.category` by delegating the conversion to scikit-learn encoder.
  :pr:`1002` by :user:`Guillaume Lemaitre <glemaitre>`.

- Handle sparse matrices in :class:`~imblearn.over_sampling.SMOTEN` and raise a warning
  since it requires a conversion to dense matrices.
  :pr:`1003` by :user:`Guillaume Lemaitre <glemaitre>`.

- Remove spurious warning raised when minority class get over-sampled more than the
  number of sample in the majority class.
  :pr:`1007` by :user:`Guillaume Lemaitre <glemaitre>`.

Compatibility
.............

- Maintenance release for being compatible with scikit-learn >= 1.3.0.
  :pr:`999` by :user:`Guillaume Lemaitre <glemaitre>`.

Deprecation
...........

- The fitted attribute `ohe_` in :class:`~imblearn.over_sampling.SMOTENC` is deprecated
  and will be removed in version 0.13. Use `categorical_encoder_` instead.
  :pr:`1000` by :user:`Guillaume Lemaitre <glemaitre>`.

- The default of the parameters `sampling_strategy`, `bootstrap` and
  `replacement` will change in
  :class:`~imblearn.ensemble.BalancedRandomForestClassifier` to follow the
  implementation of the original paper. This changes will take effect in
  version 0.13.
  :pr:`1006` by :user:`Guillaume Lemaitre <glemaitre>`.

Enhancements
............

- :class:`~imblearn.over_sampling.SMOTENC` now accepts a parameter `categorical_encoder`
  allowing to specify a :class:`~sklearn.preprocessing.OneHotEncoder` with custom
  parameters.
  :pr:`1000` by :user:`Guillaume Lemaitre <glemaitre>`.

- :class:`~imblearn.over_sampling.SMOTEN` now accepts a parameter `categorical_encoder`
  allowing to specify a :class:`~sklearn.preprocessing.OrdinalEncoder` with custom
  parameters. A new fitted parameter `categorical_encoder_` is exposed to access the
  fitted encoder.
  :pr:`1001` by :user:`Guillaume Lemaitre <glemaitre>`.

- :class:`~imblearn.under_sampling.RandomUnderSampler` and
  :class:`~imblearn.over_sampling.RandomOverSampler` (when `shrinkage is not
  None`) now accept any data types and will not attempt any data conversion.
  :pr:`1004` by :user:`Guillaume Lemaitre <glemaitre>`.

- :class:`~imblearn.over_sampling.SMOTENC` now support passing array-like of `str`
  when passing the `categorical_features` parameter.
  :pr:`1008` by :user`Guillaume Lemaitre <glemaitre>`.

- :class:`~imblearn.over_sampling.SMOTENC` now support automatic categorical inference
  when `categorical_features` is set to `"auto"`.
  :pr:`1009` by :user`Guillaume Lemaitre <glemaitre>`.
