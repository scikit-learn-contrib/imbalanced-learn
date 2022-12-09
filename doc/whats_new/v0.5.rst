.. _changes_0_5:

Version 0.5.0
=============

**June 28, 2019**

Changelog
---------

Changed models
..............

The following models or function might give different results even if the
same data ``X`` and ``y`` are the same.

* :class:`imblearn.ensemble.RUSBoostClassifier` default estimator changed from
  :class:`sklearn.tree.DecisionTreeClassifier` with full depth to a decision
  stump (i.e., tree with ``max_depth=1``).

Documentation
.............

- Correct the definition of the ratio when using a ``float`` in sampling
  strategy for the over-sampling and under-sampling.
  :issue:`525` by :user:`Ariel Rossanigo <arielrossanigo>`.

- Add :class:`imblearn.over_sampling.BorderlineSMOTE` and
  :class:`imblearn.over_sampling.SVMSMOTE` in the API documenation.
  :issue:`530` by :user:`Guillaume Lemaitre <glemaitre>`.

Enhancement
...........

- Add Parallelisation for SMOTEENN and SMOTETomek.
  :pr:`547` by :user:`Michael Hsieh <Microsheep>`.

- Add :class:`imblearn.utils._show_versions`. Updated the contribution guide
  and issue template showing how to print system and dependency information
  from the command line. :pr:`557` by :user:`Alexander L. Hayes <batflyer>`.

- Add :class:`imblearn.over_sampling.KMeansSMOTE` which is an over-sampler
  clustering points before to apply SMOTE.
  :pr:`435` by :user:`Stephan Heijl <StephanHeijl>`.

Maintenance
...........

- Make it possible to ``import imblearn`` and access submodule.
  :pr:`500` by :user:`Guillaume Lemaitre <glemaitre>`.

- Remove support for Python 2, remove deprecation warning from
  scikit-learn 0.21.
  :pr:`576` by :user:`Guillaume Lemaitre <glemaitre>`.

Bug
...

- Fix wrong usage of :class:`keras.layers.BatchNormalization` in
  ``porto_seguro_keras_under_sampling.py`` example. The batch normalization
  was moved before the activation function and the bias was removed from the
  dense layer.
  :pr:`531` by :user:`Guillaume Lemaitre <glemaitre>`.

- Fix bug which converting to COO format sparse when stacking the matrices in
  :class:`imblearn.over_sampling.SMOTENC`. This bug was only old scipy version.
  :pr:`539` by :user:`Guillaume Lemaitre <glemaitre>`.

- Fix bug in :class:`imblearn.pipeline.Pipeline` where None could be the final
  estimator.
  :pr:`554` by :user:`Oliver Rausch <orausch>`.

- Fix bug in :class:`imblearn.over_sampling.SVMSMOTE` and
  :class:`imblearn.over_sampling.BorderlineSMOTE` where the default parameter
  of ``n_neighbors`` was not set properly.
  :pr:`578` by :user:`Guillaume Lemaitre <glemaitre>`.

- Fix bug by changing the default depth in
  :class:`imblearn.ensemble.RUSBoostClassifier` to get a decision stump as a
  weak learner as in the original paper.
  :pr:`545` by :user:`Christos Aridas <chkoar>`.

- Allow to import ``keras`` directly from ``tensorflow`` in the
  :mod:`imblearn.keras`.
  :pr:`531` by :user:`Guillaume Lemaitre <glemaitre>`.
