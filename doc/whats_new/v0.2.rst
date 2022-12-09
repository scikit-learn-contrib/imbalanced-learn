.. _changes_0_2:

Version 0.2
===========

**January 1, 2017**

Changelog
---------

Bug fixes
~~~~~~~~~

- Fixed a bug in :class:`under_sampling.NearMiss` which was not picking the
  right samples during under sampling for the method 3. By :user:`Guillaume
  Lemaitre <glemaitre>`.

- Fixed a bug in :class:`ensemble.EasyEnsemble`, correction of the
  `random_state` generation. By :user:`Guillaume Lemaitre <glemaitre>` and
  :user:`Christos Aridas <chkoar>`.

- Fixed a bug in :class:`under_sampling.RepeatedEditedNearestNeighbours`, add
  additional stopping criterion to avoid that the minority class become a
  majority class or that a class disappear. By :user:`Guillaume Lemaitre
  <glemaitre>`.

- Fixed a bug in :class:`under_sampling.AllKNN`, add stopping criteria to avoid
  that the minority class become a majority class or that a class disappear. By
  :user:`Guillaume Lemaitre <glemaitre>`.

- Fixed a bug in :class:`under_sampling.CondensedNeareastNeigbour`, correction
  of the list of indices returned. By :user:`Guillaume Lemaitre <glemaitre>`.

- Fixed a bug in :class:`ensemble.BalanceCascade`, solve the issue to obtain a
  single array if desired. By :user:`Guillaume Lemaitre <glemaitre>`.

- Fixed a bug in :class:`pipeline.Pipeline`, solve to embed `Pipeline` in other
  `Pipeline`. :issue:`231` by :user:`Christos Aridas <chkoar>`.

- Fixed a bug in :class:`pipeline.Pipeline`, solve the issue to put to sampler
  in the same `Pipeline`. :issue:`188` by :user:`Christos Aridas <chkoar>`.

- Fixed a bug in :class:`under_sampling.CondensedNeareastNeigbour`, correction
  of the shape of `sel_x` when only one sample is selected. By
  :user:`Aliaksei Halachkin <honeyext>`.

- Fixed a bug in :class:`under_sampling.NeighbourhoodCleaningRule`, selecting
  neighbours instead of minority class misclassified samples. :issue:`230` by
  :user:`Aleksandr Loskutov <loskutyan>`.

- Fixed a bug in :class:`over_sampling.ADASYN`, correction of the creation of a
  new sample so that the new sample lies between the minority sample and the
  nearest neighbour. :issue:`235` by :user:`Rafael Wampfler <Eichnof>`.

New features
~~~~~~~~~~~~

- Added AllKNN under sampling technique. By :user:`Dayvid Oliveira <dvro>`.

- Added a module `metrics` implementing some specific scoring function for the
  problem of balancing. :issue:`204` by :user:`Guillaume Lemaitre <glemaitre>`
  and :user:`Christos Aridas <chkoar>`.

Enhancement
~~~~~~~~~~~

- Added support for bumpversion. By :user:`Guillaume Lemaitre <glemaitre>`.

- Validate the type of target in binary samplers. A warning is raised for the
  moment. By :user:`Guillaume Lemaitre <glemaitre>` and :user:`Christos Aridas
  <chkoar>`.

- Change from `cross_validation` module to `model_selection` module for
  `sklearn` deprecation cycle. By :user:`Dayvid Oliveira <dvro>` and
  :user:`Christos Aridas <chkoar>`.

API changes summary
~~~~~~~~~~~~~~~~~~~

- `size_ngh` has been deprecated in :class:`combine.SMOTEENN`. Use
  `n_neighbors` instead. By :user:`Guillaume Lemaitre <glemaitre>`,
  :user:`Christos Aridas <chkoar>`, and :user:`Dayvid Oliveira <dvro>`.

- `size_ngh` has been deprecated in
  :class:`under_sampling.EditedNearestNeighbors`. Use `n_neighbors` instead. By
  :user:`Guillaume Lemaitre <glemaitre>`, :user:`Christos Aridas <chkoar>`,
  and :user:`Dayvid Oliveira <dvro>`.

- `size_ngh` has been deprecated in
  :class:`under_sampling.CondensedNeareastNeigbour`. Use `n_neighbors`
  instead. By :user:`Guillaume Lemaitre <glemaitre>`,
  :user:`Christos Aridas <chkoar>`, and
  :user:`Dayvid Oliveira <dvro>`.

- `size_ngh` has been deprecated in
  :class:`under_sampling.OneSidedSelection`. Use `n_neighbors` instead. By
  :user:`Guillaume Lemaitre <glemaitre>`, :user:`Christos Aridas <chkoar>`,
  and :user:`Dayvid Oliveira <dvro>`.

- `size_ngh` has been deprecated in
  :class:`under_sampling.NeighbourhoodCleaningRule`. Use `n_neighbors`
  instead. By :user:`Guillaume Lemaitre <glemaitre>`,
  :user:`Christos Aridas <chkoar>`, and
  :user:`Dayvid Oliveira <dvro>`.

- `size_ngh` has been deprecated in
  :class:`under_sampling.RepeatedEditedNearestNeighbours`. Use `n_neighbors`
  instead. By :user:`Guillaume Lemaitre <glemaitre>`,
  :user:`Christos Aridas <chkoar>`, and
  :user:`Dayvid Oliveira <dvro>`.

- `size_ngh` has been deprecated in :class:`under_sampling.AllKNN`. Use
  `n_neighbors` instead. By :user:`Guillaume Lemaitre <glemaitre>`,
  :user:`Christos Aridas <chkoar>`, and :user:`Dayvid Oliveira <dvro>`.

- Two base classes :class:`BaseBinaryclassSampler` and
  :class:`BaseMulticlassSampler` have been created to handle the target type
  and raise warning in case of abnormality.
  By :user:`Guillaume Lemaitre <glemaitre>` and :user:`Christos Aridas <chkoar>`.

- Move `random_state` to be assigned in the :class:`SamplerMixin`
  initialization. By :user:`Guillaume Lemaitre <glemaitre>`.

- Provide estimators instead of parameters in :class:`combine.SMOTEENN` and
  :class:`combine.SMOTETomek`. Therefore, the list of parameters have been
  deprecated. By :user:`Guillaume Lemaitre <glemaitre>` and
  :user:`Christos Aridas <chkoar>`.

- `k` has been deprecated in :class:`over_sampling.ADASYN`. Use `n_neighbors`
  instead. :issue:`183` by :user:`Guillaume Lemaitre <glemaitre>`.

- `k` and `m` have been deprecated in :class:`over_sampling.SMOTE`. Use
  `k_neighbors` and `m_neighbors` instead. :issue:`182` by :user:`Guillaume
  Lemaitre <glemaitre>`.

- `n_neighbors` accept `KNeighborsMixin` based object for
  :class:`under_sampling.EditedNearestNeighbors`,
  :class:`under_sampling.CondensedNeareastNeigbour`,
  :class:`under_sampling.NeighbourhoodCleaningRule`,
  :class:`under_sampling.RepeatedEditedNearestNeighbours`, and
  :class:`under_sampling.AllKNN`. :issue:`109` by :user:`Guillaume Lemaitre
  <glemaitre>`.

Documentation changes
~~~~~~~~~~~~~~~~~~~~~

- Replace some remaining `UnbalancedDataset` occurences.
  By :user:`Francois Magimel <Linkid>`.

- Added doctest in the documentation. By :user:`Guillaume Lemaitre
  <glemaitre>`.
