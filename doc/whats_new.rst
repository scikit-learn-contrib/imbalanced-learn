.. currentmodule:: imbalanced-learn

.. _changes_0_1:

===============
Release history
===============

.. _changes_0_1:

Version 0.1
===========

Changelog
---------

Bug fixes
~~~~~~~~~

- Fixed a bug in :class:`under_sampling.NearMiss` which was not picking the right samples during under sampling for the method 3. By `Guillaume Lemaitre`_.
- Fixed a bug in :class:`ensemble.EasyEnsemble`, correction of the `random_state` generation. By `Guillaume Lemaitre`_ and `Christos Aridas`_.

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
