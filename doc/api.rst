######################
`imbalanced-learn` API
######################

This is the full API documentation of the `imbalanced-learn` toolbox.

.. _under_sampling_ref:

Under-sampling methods
======================

.. automodule:: imblearn.under_sampling
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: imblearn

.. autosummary::
   :toctree: generated/

   under_sampling.ClusterCentroids
   under_sampling.CondensedNearestNeighbour
   under_sampling.EditedNearestNeighbours
   under_sampling.RepeatedEditedNearestNeighbours
   under_sampling.AllKNN
   under_sampling.InstanceHardnessThreshold
   under_sampling.NearMiss
   under_sampling.NeighbourhoodCleaningRule
   under_sampling.OneSidedSelection
   under_sampling.RandomUnderSampler
   under_sampling.TomekLinks


.. _over_sampling_ref:

Over-sampling methods
=====================

.. automodule:: imblearn.over_sampling
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: imblearn

.. autosummary::
   :toctree: generated/

   over_sampling.ADASYN
   over_sampling.RandomOverSampler
   over_sampling.SMOTE


.. _combine_ref:

Combination of over- and under-sampling methods
===============================================

.. automodule:: imblearn.combine
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: imblearn

.. autosummary::
   :toctree: generated/

   combine.SMOTEENN
   combine.SMOTETomek


.. _ensemble_ref:

Ensemble methods
================

.. automodule:: imblearn.ensemble
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: imblearn

.. autosummary::
   :toctree: generated/

   ensemble.BalanceCascade
   ensemble.EasyEnsemble


.. _pipeline_ref:

Pipeline
========

.. automodule:: imblearn.pipeline
    :no-members:
    :no-inherited-members:

.. currentmodule:: imblearn

Classes
-------
.. autosummary::
   :toctree: generated/

   pipeline.Pipeline

Functions
---------
.. autosummary::
   :toctree: generated/

   pipeline.make_pipeline

.. _metrics_ref:

Metrics
=======

.. automodule:: imblearn.metrics
   :no-members:
   :no-inherited-members:

.. currentmodule:: imblearn

Functions
---------
.. autosummary::
   :toctree: generated/

   metrics.sensitivity_specificity_support
   metrics.sensitivity_score
   metrics.specificity_score
   metrics.geometric_mean_score
   metrics.make_index_balanced_accuracy

.. _datasets_ref:

Datasets
========

.. automodule:: imblearn.datasets
    :no-members:
    :no-inherited-members:

.. currentmodule:: imblearn

Functions
---------
.. autosummary::
   :toctree: generated/

   datasets.make_imbalance
