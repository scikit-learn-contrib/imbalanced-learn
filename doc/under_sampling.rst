.. _under-sampling:

==============
Under-sampling
==============

.. currentmodule:: imblearn.under_sampling

Prototype generation
====================

Given an original data set :math:`S`, prototype generation algorithms will
generate a new set :math:`S'` where :math:`|S'| < |S|` and :math:`S' \not\in
S`. In other words, prototype generation technique will reduce the number of
samples in the classes targeted but the remaining samples are generated --- and
not selected --- from the original set.

:class:`ClusterCentroids` makes use of K-means to reduce the number of
samples. Therefore, each class will be synthesized with the centroids of the
K-means method instead of the original samples. The figure below illustrates
such under-sampling.

..
   .. image:: ./modules/under_sampling/clustering.png
      :scale: 80
      :align: center

:class:`ClusterCentroids` offers an efficient way to represent the data cluster
with a reduced number of samples. Keep in mind that this method required that
your data are grouped into clusters. In addition, the number of centroids
should be set such that the under-sampled clusters are representative of the
original one.

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

:class:`RandomUnderSampler` is a fast and easy to balance the data by randomly
selecting a subset of data for the targeted classes.

..
   .. image:: ./modules/under_sampling/random_under_sampler.png
      :scale: 80
      :align: center

It is also possible to bootstrap the data when resampling by setting
``replacement`` to ``True``.

:class:`NearMiss`

..
   .. image:: ./modules/under_sampling/nearmiss.png
      :scale: 80
      :align: center


Cleaning under-sampling techniques
----------------------------------

:class:`InstanceHardnessThreshold`
:class:`CondensedNearestNeighbour`
:class:`EditedNearestNeighbours`
:class:`RepeatedEditedNearestNeighbours`
:class:`AllKNN`
:class:`NeighbourhoodCleaningRule`
:class:`OneSidedSelection`
:class:`TomekLinks`
