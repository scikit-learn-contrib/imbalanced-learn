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

.. image:: ./auto_examples/under-sampling/images/sphx_glr_plot_comparison_under_sampling_001.png
   :target: ./auto_examples/under-sampling/plot_comparison_under_sampling.html
   :scale: 60
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

.. image:: ./auto_examples/under-sampling/images/sphx_glr_plot_comparison_under_sampling_002.png
   :target: ./auto_examples/under-sampling/plot_comparison_under_sampling.html
   :scale: 60
   :align: center

It is also possible to bootstrap the data when resampling by setting
``replacement`` to ``True``. The resampling with multiple classes is performed
by considering independently each targeted class.

:class:`NearMiss` adds some heuristic rules to select
samples. :class:`NearMiss` implements 3 different types of heuristic which can
be selected with the parameter ``version``.

Let *positive samples* be the samples belonging to the targeted class to be
under-sampled. *Negative sample* refers to the samples from the minority class
(i.e., the most under-represented class).

NearMiss-1 selects the positive samples for which the average distance
to the :math:`N` closest samples of the negative class is the smallest.

.. image:: ./auto_examples/under-sampling/images/sphx_glr_plot_illustration_nearmiss_001.png
   :target: ./auto_examples/under-sampling/plot_illustration_nearmiss.html
   :scale: 60
   :align: center

NearMiss-2 selects the positive samples for which the average distance to the
:math:`N` farthest samples of the negative class is the smallest.

.. image:: ./auto_examples/under-sampling/images/sphx_glr_plot_illustration_nearmiss_002.png
   :target: ./auto_examples/under-sampling/plot_illustration_nearmiss.html
   :scale: 60
   :align: center

NearMiss-3 is a 2-steps algorithm. First, for each negative sample, their
:math:`M` nearest-neighbors will be kept. Then, the positive samples selected
are the one for which the average distance to the :math:`N` nearest-neighbors
is the largest.

.. image:: ./auto_examples/under-sampling/images/sphx_glr_plot_illustration_nearmiss_003.png
   :target: ./auto_examples/under-sampling/plot_illustration_nearmiss.html
   :scale: 60
   :align: center

In the next example, the different :class:`NearMiss` variant are applied on the
previous toy example. It can be seen that the decision functions obtained in
each case are different.

When under-sampling a specific class, NearMiss-1 can be altered by the presence
of noise. In fact, it will implied that samples of the targeted class will be
selected around these samples as it is the case in the illustration below for
the yellow class. However, in the normal case, samples next to the boundaries
will be selected. NearMiss-2 will not have this effect since it does not focus
on the nearest samples but rather on the farthest samples. We can imagine that
the presence of noise can also altered the sampling mainly in the presence of
marginal outliers. NearMiss-3 is probably the version which will be the less
affected by noise due to the first step sample selection.

.. image:: ./auto_examples/under-sampling/images/sphx_glr_plot_comparison_under_sampling_003.png
   :target: ./auto_examples/under-sampling/plot_comparison_under_sampling.html
   :scale: 60
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
