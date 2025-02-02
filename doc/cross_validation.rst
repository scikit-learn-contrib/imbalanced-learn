.. _cross_validation:

================
Cross validation
================

.. currentmodule:: imblearn.cross_validation


.. _instance_hardness_threshold:

The term instance hardness is used in literature to express the difficulty to
correctly classify an instance. An instance for which the predicted probability
of the true class is low, has large instance hardness. The way these
hard-to-classify instances are distributed over train and test sets in cross
validation, has significant effect on the test set performance metrics. The
`InstanceHardnessCV` splitter distributes samples with large instance hardness
equally over the folds, resulting in more robust cross validation.
