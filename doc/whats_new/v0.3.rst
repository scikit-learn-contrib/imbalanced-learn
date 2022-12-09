.. _changes_0_3:

Version 0.3
===========

**February 22, 2018**

Changelog
---------

Testing
~~~~~~~
- Pytest is used instead of nosetests. :issue:`321` by :user:`Joan Massich
  <massich>`.

Documentation
~~~~~~~~~~~~~

- Added a User Guide and extended some examples. :issue:`295` by
  :user:`Guillaume Lemaitre <glemaitre>`.

Bug fixes
~~~~~~~~~

- Fixed a bug in :func:`utils.check_ratio` such that an error is raised when
  the number of samples required is negative. :issue:`312` by :user:`Guillaume
  Lemaitre <glemaitre>`.

- Fixed a bug in :class:`under_sampling.NearMiss` version 3. The indices
  returned were wrong. :issue:`312` by :user:`Guillaume Lemaitre <glemaitre>`.

- Fixed bug for :class:`ensemble.BalanceCascade` and :class:`combine.SMOTEENN`
  and :class:`SMOTETomek`. :issue:`295` by :user:`Guillaume Lemaitre
  <glemaitre>`.

- Fixed bug for `check_ratio` to be able to pass arguments when `ratio` is a
  callable. :issue:`307` by :user:`Guillaume Lemaitre <glemaitre>`.

New features
~~~~~~~~~~~~

- Turn off steps in :class:`pipeline.Pipeline` using the `None`
  object. By :user:`Christos Aridas <chkoar>`.

- Add a fetching function :func:`datasets.fetch_datasets` in order to get some
  imbalanced datasets useful for benchmarking. :issue:`249` by :user:`Guillaume
  Lemaitre <glemaitre>`.

Enhancement
~~~~~~~~~~~

- All samplers accepts sparse matrices with defaulting on CSR
  type. :issue:`316` by :user:`Guillaume Lemaitre <glemaitre>`.

- :func:`datasets.make_imbalance` take a ratio similarly to other samplers. It
  supports multiclass. :issue:`312` by :user:`Guillaume Lemaitre <glemaitre>`.

- All the unit tests have been factorized and a :func:`utils.check_estimators`
  has been derived from scikit-learn. By :user:`Guillaume Lemaitre
  <glemaitre>`.

- Script for automatic build of conda packages and uploading. :issue:`242` by
  :user:`Guillaume Lemaitre <glemaitre>`

- Remove seaborn dependence and improve the examples. :issue:`264` by
  :user:`Guillaume Lemaitre <glemaitre>`.

- adapt all classes to multi-class resampling. :issue:`290` by :user:`Guillaume
  Lemaitre <glemaitre>`

API changes summary
~~~~~~~~~~~~~~~~~~~

- `__init__` has been removed from the :class:`base.SamplerMixin` to create a
  real mixin class. :issue:`242` by :user:`Guillaume Lemaitre <glemaitre>`.

- creation of a module :mod:`exceptions` to handle consistant raising of
  errors. :issue:`242` by :user:`Guillaume Lemaitre <glemaitre>`.

- creation of a module ``utils.validation`` to make checking of recurrent
  patterns. :issue:`242` by :user:`Guillaume Lemaitre <glemaitre>`.

- move the under-sampling methods in ``prototype_selection`` and
  ``prototype_generation`` submodule to make a clearer
  dinstinction. :issue:`277` by :user:`Guillaume Lemaitre <glemaitre>`.

- change ``ratio`` such that it can adapt to multiple class
  problems. :issue:`290` by :user:`Guillaume Lemaitre <glemaitre>`.

Deprecation
~~~~~~~~~~~

- Deprecation of the use of ``min_c_`` in
  :func:`datasets.make_imbalance`. :issue:`312` by :user:`Guillaume Lemaitre
  <glemaitre>`

- Deprecation of the use of float in :func:`datasets.make_imbalance` for the
  ratio parameter. :issue:`290` by :user:`Guillaume Lemaitre <glemaitre>`.

- deprecate the use of float as ratio in favor of dictionary, string, or
  callable. :issue:`290` by :user:`Guillaume Lemaitre <glemaitre>`.
