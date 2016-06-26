#################
API Documentation
#################

This is the full API documentation of the `unbalanced_dataset` toolbox.

.. _under_sampling_ref:

Under-sampling methods
======================

.. automodule:: unbalanced_dataset.under_sampling
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: unbalanced_dataset

.. autosummary::
   :toctree: generated/
   
   under_sampling.ClusterCentroids
   under_sampling.CondensedNearestNeighbour
   under_sampling.EditedNearestNeighbours
   under_sampling.RepeatedEditedNearestNeighbours
   under_sampling.InstanceHardnessThreshold
   under_sampling.NearMiss
   under_sampling.NeighbourhoodCleaningRule
   under_sampling.OneSidedSelection
   under_sampling.RandomUnderSampler
   under_sampling.TomekLinks


.. _over_sampling_ref:

Over-sampling methods
=====================

.. automodule:: unbalanced_dataset.over_sampling
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: unbalanced_dataset

.. autosummary::
   :toctree: generated/
   
   over_sampling.RandomOverSampler
   over_sampling.SMOTE


.. _combine_ref:

Combination of over- and under-sampling methods
===============================================

.. automodule:: unbalanced_dataset.combine
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: unbalanced_dataset

.. autosummary::
   :toctree: generated/
   
   combine.SMOTEENN
   combine.SMOTETomek


.. _ensemble_ref:

Ensemble methods
================

.. automodule:: unbalanced_dataset.ensemble
    :no-members:
    :no-inherited-members:

Classes
-------
.. currentmodule:: unbalanced_dataset

.. autosummary::
   :toctree: generated/
   
   ensemble.BalanceCascade
   ensemble.EasyEnsemble


.. _pipeline_ref:

Pipeline
========

.. automodule:: unbalanced_dataset.pipeline
    :no-members:
    :no-inherited-members:

.. currentmodule:: unbalanced_dataset

.. autosummary::
   :toctree: generated/
   
   pipeline.Pipeline

.. autosummary::
   :toctree: generated/
   
   pipeline.make_pipeline
