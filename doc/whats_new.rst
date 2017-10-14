.. currentmodule:: imblearn

===============
Release history
===============

.. _changes_0_3:

Version 0.3
===========

Changelog
---------

Testing
~~~~~~~

- Pytest is used instead of nosetests. By `Joan Massich`_.

Documentation
~~~~~~~~~~~~~

- Added a User Guide and extended some examples. By `Guillaume Lemaitre`_.

Bug fixes
~~~~~~~~~

- Fixed a bug in :func:`utils.check_ratio` such that an error is raised when
  the number of samples required is negative. By `Guillaume Lemaitre`_.

- Fixed a bug in :class:`under_sampling.NearMiss` version 3. The
  indices returned were wrong. By `Guillaume Lemaitre`_.

- Fixed bug for :class:`ensemble.BalanceCascade` and :class:`combine.SMOTEENN`
  and :class:`SMOTETomek`. By `Guillaume Lemaitre`_.`

- Fixed bug for `check_ratio` to be able to pass arguments when `ratio` is a
  callable. By `Guillaume Lemaitre`_.`

- Fix bug in ADASYN to consider only samples from the current class when
  generating new samples. :issue:`354` by :user:`Guillaume Lemaitre
  <glemaitre>`.

New features
~~~~~~~~~~~~

- :class:`under_sampling.ClusterCentroids` accepts a parameter ``voting``
  allowing to use nearest-neighbors of centroids instead of centroids
  themselves. It is more efficient for sparse input. By `Guillaume Lemaitre`_.

- Turn off steps in :class:`pipeline.Pipeline` using the `None`
  object. By `Christos Aridas`_.

- Add a fetching function :func:`datasets.fetch_datasets` in order to get some
  imbalanced datasets useful for benchmarking. By `Guillaume Lemaitre`_.

Enhancement
~~~~~~~~~~~

- Add :class:`ensemble.BalancedBaggingClassifier` which is a meta estimator to
  directly use the :class:`ensemble.EasyEnsemble` chained with a classifier. By
  `Guillaume Lemaitre`_.

- All samplers accepts sparse matrices with defaulting on CSR type. By
  `Guillaume Lemaitre`_.

- :func:`datasets.make_imbalance` take a ratio similarly to other samplers. It
  supports multiclass. By `Guillaume Lemaitre`_.

- All the unit tests have been factorized and a :func:`utils.check_estimators`
  has been derived from scikit-learn. By `Guillaume Lemaitre`_.

- Script for automatic build of conda packages and uploading. By
  `Guillaume Lemaitre`_

- Remove seaborn dependence and improve the examples. By `Guillaume
  Lemaitre`_.

- adapt all classes to multi-class resampling. By `Guillaume Lemaitre`_

API changes summary
~~~~~~~~~~~~~~~~~~~

- `__init__` has been removed from the :class:`base.SamplerMixin` to
  create a real mixin class. By `Guillaume Lemaitre`_.

- creation of a module :mod:`exceptions` to handle consistant raising of
  errors. By `Guillaume Lemaitre`_.

- creation of a module ``utils.validation`` to make checking of
  recurrent patterns. By `Guillaume Lemaitre`_.

- move the under-sampling methods in ``prototype_selection`` and
  ``prototype_generation`` submodule to make a clearer dinstinction. By
  `Guillaume Lemaitre`_.

- change ``ratio`` such that it can adapt to multiple class problems. By
  `Guillaume Lemaitre`_.

Deprecation
~~~~~~~~~~~

- Deprecation of the use of ``min_c_`` in :func:`datasets.make_imbalance`. By
  `Guillaume Lemaitre`_

- Deprecation of the use of float in :func:`datasets.make_imbalance` for the
  ratio parameter. By `Guillaume Lemaitre`_.

- deprecate the use of float as ratio in favor of dictionary, string, or
  callable. By `Guillaume Lemaitre`_.

.. _changes_0_2:

Version 0.2
===========

Changelog
---------

Bug fixes
~~~~~~~~~

- Fixed a bug in :class:`under_sampling.NearMiss` which was not picking the right samples during under sampling for the method 3. By `Guillaume Lemaitre`_.
- Fixed a bug in :class:`ensemble.EasyEnsemble`, correction of the `random_state` generation. By `Guillaume Lemaitre`_ and `Christos Aridas`_.
- Fixed a bug in :class:`under_sampling.RepeatedEditedNearestNeighbours`, add additional stopping criterion to avoid that the minority class become a majority class or that a class disappear. By `Guillaume Lemaitre`_.
- Fixed a bug in :class:`under_sampling.AllKNN`, add stopping criteria to avoid that the minority class become a majority class or that a class disappear. By `Guillaume Lemaitre`_.
- Fixed a bug in :class:`under_sampling.CondensedNeareastNeigbour`, correction of the list of indices returned. By `Guillaume Lemaitre`_.
- Fixed a bug in :class:`ensemble.BalanceCascade`, solve the issue to obtain a single array if desired. By `Guillaume Lemaitre`_.
- Fixed a bug in :class:`pipeline.Pipeline`, solve to embed `Pipeline` in other `Pipeline`. By `Christos Aridas`_ .
- Fixed a bug in :class:`pipeline.Pipeline`, solve the issue to put to sampler in the same `Pipeline`. By `Christos Aridas`_ .
- Fixed a bug in :class:`under_sampling.CondensedNeareastNeigbour`, correction of the shape of `sel_x` when only one sample is selected. By `Aliaksei Halachkin`_.
- Fixed a bug in :class:`under_sampling.NeighbourhoodCleaningRule`, selecting neighbours instead of minority class misclassified samples. By `Aleksandr Loskutov`_.
- Fixed a bug in :class:`over_sampling.ADASYN`, correction of the creation of a new sample so that the new sample lies between the minority sample and the nearest neighbour. By `Rafael Wampfler`_.

New features
~~~~~~~~~~~~

- Added AllKNN under sampling technique. By `Dayvid Oliveira`_.
- Added a module `metrics` implementing some specific scoring function for the problem of balancing. By `Guillaume Lemaitre`_ and `Christos Aridas`_.

Enhancement
~~~~~~~~~~~

- Added support for bumpversion. By `Guillaume Lemaitre`_.
- Validate the type of target in binary samplers. A warning is raised for the moment. By `Guillaume Lemaitre`_ and `Christos Aridas`_.
- Change from `cross_validation` module to `model_selection` module for
  `sklearn` deprecation cycle. By `Dayvid Oliveira`_ and `Christos Aridas`_.

API changes summary
~~~~~~~~~~~~~~~~~~~

- `size_ngh` has been deprecated in :class:`combine.SMOTEENN`. Use `n_neighbors` instead. By `Guillaume Lemaitre`_, `Christos Aridas`_, and `Dayvid Oliveira` .
- `size_ngh` has been deprecated in :class:`under_sampling.EditedNearestNeighbors`. Use `n_neighbors` instead. By `Guillaume Lemaitre`_, `Christos Aridas`_, and `Dayvid Oliveira`_.
- `size_ngh` has been deprecated in :class:`under_sampling.CondensedNeareastNeigbour`. Use `n_neighbors` instead. By `Guillaume Lemaitre`_, `Christos Aridas`_, and `Dayvid Oliveira`_.
- `size_ngh` has been deprecated in :class:`under_sampling.OneSidedSelection`. Use `n_neighbors` instead. By `Guillaume Lemaitre`_, `Christos Aridas`_, and `Dayvid Oliveira`_.
- `size_ngh` has been deprecated in :class:`under_sampling.NeighbourhoodCleaningRule`. Use `n_neighbors` instead. By `Guillaume Lemaitre`_, `Christos Aridas`_, and `Dayvid Oliveira`_.
- `size_ngh` has been deprecated in :class:`under_sampling.RepeatedEditedNearestNeighbours`. Use `n_neighbors` instead. By `Guillaume Lemaitre`_, `Christos Aridas`_, and `Dayvid Oliveira`_.
- `size_ngh` has been deprecated in :class:`under_sampling.AllKNN`. Use `n_neighbors` instead. By `Guillaume Lemaitre`_, `Christos Aridas`_, and `Dayvid Oliveira`_.
- Two base classes :class:`BaseBinaryclassSampler` and :class:`BaseMulticlassSampler` have been created to handle the target type and raise warning in case of abnormality. By `Guillaume Lemaitre`_ and `Christos Aridas`_.
- Move `random_state` to be assigned in the :class:`SamplerMixin` initialization. By `Guillaume Lemaitre`_.
- Provide estimators instead of parameters in :class:`combine.SMOTEENN` and :class:`combine.SMOTETomek`. Therefore, the list of parameters have been deprecated. By `Guillaume Lemaitre`_ and `Christos Aridas`_.
- `k` has been deprecated in :class:`over_sampling.ADASYN`. Use `n_neighbors` instead. By `Guillaume Lemaitre`_.
- `k` and `m` have been deprecated in :class:`over_sampling.SMOTE`. Use `k_neighbors` and `m_neighbors` instead. By `Guillaume Lemaitre`_.
- `n_neighbors` accept `KNeighborsMixin` based object for :class:`under_sampling.EditedNearestNeighbors`, :class:`under_sampling.CondensedNeareastNeigbour`, :class:`under_sampling.NeighbourhoodCleaningRule`, :class:`under_sampling.RepeatedEditedNearestNeighbours`, and :class:`under_sampling.AllKNN`. By `Guillaume Lemaitre`_.

Documentation changes
~~~~~~~~~~~~~~~~~~~~~

- Replace some remaining `UnbalancedDataset` occurences. By `Francois Magimel`_.
- Added doctest in the documentation. By `Guillaume Lemaitre`_.

.. _changes_0_1:

Version 0.1
===========

Changelog
---------

API
~~~

- First release of the stable API. By `Fernando Nogueira`_, `Guillaume Lemaitre`_, `Christos Aridas`_, and `Dayvid Oliveira`_.

New methods
~~~~~~~~~~~

* Under-sampling
    1. Random majority under-sampling with replacement
    2. Extraction of majority-minority Tomek links
    3. Under-sampling with Cluster Centroids
    4. NearMiss-(1 & 2 & 3)
    5. Condensend Nearest Neighbour
    6. One-Sided Selection
    7. Neighboorhood Cleaning Rule
    8. Edited Nearest Neighbours
    9. Instance Hardness Threshold
    10. Repeated Edited Nearest Neighbours

* Over-sampling
    1. Random minority over-sampling with replacement
    2. SMOTE - Synthetic Minority Over-sampling Technique
    3. bSMOTE(1 & 2) - Borderline SMOTE of types 1 and 2
    4. SVM SMOTE - Support Vectors SMOTE
    5. ADASYN - Adaptive synthetic sampling approach for imbalanced learning

* Over-sampling followed by under-sampling
    1. SMOTE + Tomek links
    2. SMOTE + ENN

* Ensemble sampling
    1. EasyEnsemble
    2. BalanceCascade

.. _Guillaume Lemaitre: https://github.com/glemaitre
.. _Christos Aridas: https://github.com/chkoar
.. _Fernando Nogueira: https://github.com/fmfn
.. _Dayvid Oliveira: https://github.com/dvro
.. _Francois Magimel: https://github.com/Linkid
.. _Aliaksei Halachkin: https://github.com/honeyext
.. _Aleksandr Loskutov: https://github.com/loskutyan
.. _Rafael Wampfler: https://github.com/Eichhof
.. _Joan Massich: https://github.com/massich
