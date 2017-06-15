.. _under-sampling:

==============
Under-sampling
==============

.. currentmodule:: imblearn.under_sampling

Prototype generation
====================

Given an original data set :math:`S`, prototype generation algorithms will
generate a new set :math:`S'` where :math:`|S'| < |S|` and :math:`S' \not\in
S`.

:class:`ClusterCentroids`

Prototype selection
===================

On the contrary to prototype generation algorithms, prototype selection
algorithms will select samples from the original set :math:`S`. Therefore,
:math:`S'` is defined such as :math:`|S'| < |S|` and :math:`S' \in S`.

In addition, these algorithms can be divided into two groups: (i) the
controlled under-sampling techniques and (ii) the cleaning under-sampling
techniques. The first group of methods allows for an under-sampling strategy in
which the number of samples in :math:`S'` is specified by the user. By
contrast, cleaning under-sampling techniques do not allow this specification
and are meant for cleaning the feature space.

Controlled under-sampling techniques
------------------------------------

:class:`NearMiss`
:class:`InstanceHardnessThreshold`
:class:`RandomUnderSampler`

Cleaning under-sampling techniques
----------------------------------

:class:`CondensedNearestNeighbour`
:class:`EditedNearestNeighbours`
:class:`RepeatedEditedNearestNeighbours`
:class:`AllKNN`
:class:`NeighbourhoodCleaningRule`
:class:`OneSidedSelection`
:class:`TomekLinks`
