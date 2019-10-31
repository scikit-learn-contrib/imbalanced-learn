######################
imbalanced-learn API
######################

This is the full API documentation of the `imbalanced-learn` toolbox.

.. _under_sampling_ref:

:mod:`imblearn.under_sampling`: Under-sampling methods
======================================================

.. automodule:: imblearn.under_sampling
    :no-members:
    :no-inherited-members:

.. currentmodule:: imblearn

Prototype generation
--------------------

.. automodule:: imblearn.under_sampling._prototype_generation
   :no-members:
   :no-inherited-members:

.. currentmodule:: imblearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   under_sampling.ClusterCentroids

Prototype selection
-------------------

.. automodule:: imblearn.under_sampling._prototype_selection
   :no-members:
   :no-inherited-members:

.. currentmodule:: imblearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

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

:mod:`imblearn.over_sampling`: Over-sampling methods
====================================================

.. automodule:: imblearn.over_sampling
    :no-members:
    :no-inherited-members:

.. currentmodule:: imblearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   over_sampling.ADASYN
   over_sampling.BorderlineSMOTE
   over_sampling.KMeansSMOTE
   over_sampling.RandomOverSampler
   over_sampling.SMOTE
   over_sampling.SMOTENC
   over_sampling.SVMSMOTE


.. _combine_ref:

:mod:`imblearn.combine`: Combination of over- and under-sampling methods
========================================================================

.. automodule:: imblearn.combine
   :no-members:
   :no-inherited-members:

.. currentmodule:: imblearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   combine.SMOTEENN
   combine.SMOTETomek

.. _ensemble_ref:

:mod:`imblearn.ensemble`: Ensemble methods
==========================================

.. automodule:: imblearn.ensemble
    :no-members:
    :no-inherited-members:

.. currentmodule:: imblearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   ensemble.BalancedBaggingClassifier
   ensemble.BalancedRandomForestClassifier
   ensemble.EasyEnsembleClassifier
   ensemble.RUSBoostClassifier

.. _keras_ref:

:mod:`imblearn.keras`: Batch generator for Keras
================================================

.. automodule:: imblearn.keras
    :no-members:
    :no-inherited-members:

.. currentmodule:: imblearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   keras.BalancedBatchGenerator

.. autosummary::
   :toctree: generated/
   :template: function.rst

   keras.balanced_batch_generator

.. _tensorflow_ref:

:mod:`imblearn.tensorflow`: Batch generator for TensorFlow
==========================================================

.. automodule:: imblearn.tensorflow
    :no-members:
    :no-inherited-members:

.. currentmodule:: imblearn

.. autosummary::
   :toctree: generated/
   :template: function.rst
              
   tensorflow.balanced_batch_generator

.. _misc_ref:
   
Miscellaneous
=============

Imbalance-learn provides some fast-prototyping tools.

.. currentmodule:: imblearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   FunctionSampler

.. _pipeline_ref:

:mod:`imblearn.pipeline`: Pipeline
==================================

.. automodule:: imblearn.pipeline
    :no-members:
    :no-inherited-members:

.. currentmodule:: imblearn

.. autosummary::
   :toctree: generated/
   :template: class.rst

   pipeline.Pipeline

.. autosummary::
   :toctree: generated/
   :template: function.rst

   pipeline.make_pipeline

.. _metrics_ref:

:mod:`imblearn.metrics`: Metrics
================================

.. automodule:: imblearn.metrics
   :no-members:
   :no-inherited-members:

.. currentmodule:: imblearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   metrics.classification_report_imbalanced
   metrics.sensitivity_specificity_support
   metrics.sensitivity_score
   metrics.specificity_score
   metrics.geometric_mean_score
   metrics.make_index_balanced_accuracy

.. _datasets_ref:

:mod:`imblearn.datasets`: Datasets
==================================

.. automodule:: imblearn.datasets
    :no-members:
    :no-inherited-members:

.. currentmodule:: imblearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   datasets.make_imbalance
   datasets.fetch_datasets

:mod:`imblearn.utils`: Utilities
================================

.. automodule:: imblearn.utils
    :no-members:
    :no-inherited-members:

.. currentmodule:: imblearn

.. autosummary::
   :toctree: generated/
   :template: function.rst

   utils.estimator_checks.check_estimator
   utils.check_neighbors_object
   utils.check_sampling_strategy
