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

.. autosummary::
   :toctree: generated/
   
   unbalanced_dataset.under_sampling.ClusterCentroids
   unbalanced_dataset.under_sampling.CondensedNearestNeighbour
   unbalanced_dataset.under_sampling.EditedNearestNeighbours
   unbalanced_dataset.under_sampling.RepeatedEditedNearestNeighbours
   unbalanced_dataset.under_sampling.InstanceHardnessThreshold
   unbalanced_dataset.under_sampling.NearMiss
   unbalanced_dataset.under_sampling.NeighbourhoodCleaningRule
   unbalanced_dataset.under_sampling.OneSidedSelection
   unbalanced_dataset.under_sampling.RandomUnderSampler
   unbalanced_dataset.under_sampling.TomekLinks


.. _over_sampling_ref:

Over-sampling methods
=====================

.. automodule:: unbalanced_dataset.over_sampling
    :no-members:
    :no-inherited-members:

Classes
-------

.. autosummary::
   :toctree: generated/
   
   unbalanced_dataset.over_sampling.RandomOverSampler
   unbalanced_dataset.over_sampling.SMOTE


.. _combine_ref:

Combination of over- and under-sampling methods
===============================================

.. automodule:: unbalanced_dataset.combine
    :no-members:
    :no-inherited-members:

Classes
-------

.. autosummary::
   :toctree: generated/
   
   unbalanced_dataset.combine.SMOTEENN
   unbalanced_dataset.combine.SMOTETomek

.. _ensemble_ref:

Ensemble methods
================

.. automodule:: unbalanced_dataset.ensemble
    :no-members:
    :no-inherited-members:

Classes
-------

.. autosummary::
   :toctree: generated/
   
   unbalanced_dataset.ensemble.BalanceCascade
   unbalanced_dataset.ensemble.EasyEnsemble
