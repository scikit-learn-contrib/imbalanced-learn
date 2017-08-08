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
K-means method instead of the original samples::

  >>> from collections import Counter
  >>> from sklearn.datasets import make_classification
  >>> X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
  ...                            n_redundant=0, n_repeated=0, n_classes=3,
  ...                            n_clusters_per_class=1,
  ...                            weights=[0.01, 0.05, 0.94],
  ...                            class_sep=0.8, random_state=0)
  >>> print(Counter(y))
  Counter({2: 4674, 1: 262, 0: 64})
  >>> from imblearn.under_sampling import ClusterCentroids
  >>> cc = ClusterCentroids(random_state=0)
  >>> X_resampled, y_resampled = cc.fit_sample(X, y)
  >>> print(Counter(y_resampled))
  Counter({0: 64, 1: 64, 2: 64})


The figure below illustrates such under-sampling.

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
selecting a subset of data for the targeted classes::

  >>> from imblearn.under_sampling import RandomUnderSampler
  >>> rus = RandomUnderSampler(random_state=0)
  >>> X_resampled, y_resampled = rus.fit_sample(X, y)
  >>> print(Counter(y_resampled))
  Counter({0: 64, 1: 64, 2: 64})

.. image:: ./auto_examples/under-sampling/images/sphx_glr_plot_comparison_under_sampling_002.png
   :target: ./auto_examples/under-sampling/plot_comparison_under_sampling.html
   :scale: 60
   :align: center

It is also possible to bootstrap the data when resampling by setting
``replacement`` to ``True``. The resampling with multiple classes is performed
by considering independently each targeted class::

  >>> import numpy as np
  >>> print(np.unique(X_resampled, axis=0).shape)
  (192, 2)
  >>> rus = RandomUnderSampler(random_state=0, replacement=True)
  >>> X_resampled, y_resampled = rus.fit_sample(X, y)
  >>> print(np.unique(X_resampled, axis=0).shape)
  (181, 2)

:class:`NearMiss` adds some heuristic rules to select
samples. :class:`NearMiss` implements 3 different types of heuristic which can
be selected with the parameter ``version``::

  >>> from imblearn.under_sampling import NearMiss
  >>> nm1 = NearMiss(random_state=0, version=1)
  >>> X_resampled_nm1, y_resampled = nm1.fit_sample(X, y)
  >>> print(Counter(y_resampled))
  Counter({0: 64, 1: 64, 2: 64})

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

In cleaning under-sampling techniques do not allow to specify the number
samples to have in each class. In fact, each algorithm implement an heuristic
which will clean the dataset.

Edited data set using nearest neighbours
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

:class:`EditedNearestNeighbours` applies a nearest-neighbors algorithm and
"edit" the dataset by removing samples which do not agree "enough" with their
neighboorhood. For each sample in the class to be under-sampled, the
nearest-neighbours are computed and if the selection criterion is not
fulfilled, the sample is removed. Two selection criteria are currently
available: (i) the majority (i.e., ``kind_sel='mode'``) or (ii) all (i.e.,
``kind_sel='all'``) the nearest-neighbors have to belong to the same class than
the sample inspected to keep it in the dataset::

  >>> Counter(y)
  Counter({2: 4674, 1: 262, 0: 64})
  >>> from imblearn.under_sampling import EditedNearestNeighbours
  >>> enn = EditedNearestNeighbours(random_state=0)
  >>> X_resampled, y_resampled = enn.fit_sample(X, y)
  >>> print(Counter(y_resampled))
  Counter({2: 4568, 1: 213, 0: 64})

:class:`RepeatedEditedNearestNeighbours` extends
:class:`EditedNearestNeighbours` by repeating the algorithm multiple times.
Generally, repeating the algorithm will delete more data::

   >>> from imblearn.under_sampling import RepeatedEditedNearestNeighbours
   >>> renn = RepeatedEditedNearestNeighbours(random_state=0)
   >>> X_resampled, y_resampled = renn.fit_sample(X, y)
   >>> print(Counter(y_resampled))
   Counter({2: 4551, 1: 208, 0: 64})

:class:`AllKNN` differs from the previous
:class:`RepeatedEditedNearestNeighbours` since the number of neighbors of the
internal nearest neighbors algorithm is increased at each iteration::

  >>> from imblearn.under_sampling import AllKNN
  >>> allknn = AllKNN(random_state=0)
  >>> X_resampled, y_resampled = allknn.fit_sample(X, y)
  >>> print(Counter(y_resampled))
  Counter({2: 4601, 1: 220, 0: 64})

In the example below, it can be seen that the three algorithms have similar
impact by cleaning noisy samples next to the boundaries of the classes.

.. image:: ./auto_examples/under-sampling/images/sphx_glr_plot_comparison_under_sampling_004.png
   :target: ./auto_examples/under-sampling/plot_comparison_under_sampling.html
   :scale: 60
   :align: center



:class:`InstanceHardnessThreshold`

:class:`CondensedNearestNeighbour`

:class:`NeighbourhoodCleaningRule`
:class:`OneSidedSelection`


:class:`TomekLinks`
