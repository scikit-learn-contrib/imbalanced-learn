.. _over-sampling:

=============
Over-sampling
=============

.. currentmodule:: imblearn.over_sampling

A practical guide
=================

You can refer to
:ref:`sphx_glr_auto_examples_over-sampling_plot_comparison_over_sampling.py`.

.. _random_over_sampler:

Naive random over-sampling
--------------------------

One way to fight this issue is to generate new samples in the classes which are
under-represented. The most naive strategy is to generate new samples by
randomly sampling with replacement the current available samples. The
:class:`RandomOverSampler` offers such scheme::

   >>> from sklearn.datasets import make_classification
   >>> X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
   ...                            n_redundant=0, n_repeated=0, n_classes=3,
   ...                            n_clusters_per_class=1,
   ...                            weights=[0.01, 0.05, 0.94],
   ...                            class_sep=0.8, random_state=0)
   >>> from imblearn.over_sampling import RandomOverSampler
   >>> ros = RandomOverSampler(random_state=0)
   >>> X_resampled, y_resampled = ros.fit_resample(X, y)
   >>> from collections import Counter
   >>> print(sorted(Counter(y_resampled).items()))
   [(0, 4674), (1, 4674), (2, 4674)]

The augmented data set should be used instead of the original data set to train
a classifier::

  >>> from sklearn.svm import LinearSVC
  >>> clf = LinearSVC()
  >>> clf.fit(X_resampled, y_resampled) # doctest : +ELLIPSIS
  LinearSVC(...)

In the figure below, we compare the decision functions of a classifier trained
using the over-sampled data set and the original data set.

.. image:: ./auto_examples/over-sampling/images/sphx_glr_plot_comparison_over_sampling_002.png
   :target: ./auto_examples/over-sampling/plot_comparison_over_sampling.html
   :scale: 60
   :align: center

As a result, the majority class does not take over the other classes during the
training process. Consequently, all classes are represented by the decision
function.

In addition, :class:`RandomOverSampler` allows to sample heterogeneous data
(e.g. containing some strings)::

  >>> import numpy as np
  >>> X_hetero = np.array([['xxx', 1, 1.0], ['yyy', 2, 2.0], ['zzz', 3, 3.0]],
  ...                     dtype=np.object)
  >>> y_hetero = np.array([0, 0, 1])
  >>> X_resampled, y_resampled = ros.fit_resample(X_hetero, y_hetero)
  >>> print(X_resampled)
  [['xxx' 1 1.0]
   ['yyy' 2 2.0]
   ['zzz' 3 3.0]
   ['zzz' 3 3.0]]
  >>> print(y_resampled)
  [0 0 1 1]

.. _smote_adasyn:

From random over-sampling to SMOTE and ADASYN
---------------------------------------------

Apart from the random sampling with replacement, there are two popular methods
to over-sample minority classes: (i) the Synthetic Minority Oversampling
Technique (SMOTE) [CBHK2002]_ and (ii) the Adaptive Synthetic (ADASYN)
[HBGL2008]_ sampling method. These algorithms can be used in the same manner::

  >>> from imblearn.over_sampling import SMOTE, ADASYN
  >>> X_resampled, y_resampled = SMOTE().fit_resample(X, y)
  >>> print(sorted(Counter(y_resampled).items()))
  [(0, 4674), (1, 4674), (2, 4674)]
  >>> clf_smote = LinearSVC().fit(X_resampled, y_resampled)
  >>> X_resampled, y_resampled = ADASYN().fit_resample(X, y)
  >>> print(sorted(Counter(y_resampled).items()))
  [(0, 4673), (1, 4662), (2, 4674)]
  >>> clf_adasyn = LinearSVC().fit(X_resampled, y_resampled)

The figure below illustrates the major difference of the different
over-sampling methods.

.. image:: ./auto_examples/over-sampling/images/sphx_glr_plot_comparison_over_sampling_003.png
   :target: ./auto_examples/over-sampling/plot_comparison_over_sampling.html
   :scale: 60
   :align: center

.. topic:: References

  .. [HBGL2008] He, Haibo, Yang Bai, Edwardo A. Garcia, and Shutao Li. "ADASYN:
                Adaptive synthetic sampling approach for imbalanced learning,"
                In IEEE International Joint Conference on Neural Networks (IEEE
                World Congress on Computational Intelligence), pp. 1322-1328,
                2008.

  .. [CBHK2002] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer,
                "SMOTE: synthetic minority over-sampling technique," Journal of
                artificial intelligence research, 321-357, 2002.

Ill-posed examples
------------------

While the :class:`RandomOverSampler` is over-sampling by duplicating some of
the original samples of the minority class, :class:`SMOTE` and :class:`ADASYN`
generate new samples in by interpolation. However, the samples used to
interpolate/generate new synthetic samples differ. In fact, :class:`ADASYN`
focuses on generating samples next to the original samples which are wrongly
classified using a k-Nearest Neighbors classifier while the basic
implementation of :class:`SMOTE` will not make any distinction between easy and
hard samples to be classified using the nearest neighbors rule. Therefore, the
decision function found during training will be different among the algorithms.

.. image:: ./auto_examples/over-sampling/images/sphx_glr_plot_comparison_over_sampling_004.png
   :target: ./auto_examples/over-sampling/plot_comparison_over_sampling.html
   :align: center

The sampling particularities of these two algorithms can lead to some peculiar
behavior as shown below.

.. image:: ./auto_examples/over-sampling/images/sphx_glr_plot_comparison_over_sampling_005.png
   :target: ./auto_examples/over-sampling/plot_comparison_over_sampling.html
   :scale: 60
   :align: center

SMOTE variants
--------------

SMOTE might connect inliers and outliers while ADASYN might focus solely on
outliers which, in both cases, might lead to a sub-optimal decision
function. In this regard, SMOTE offers three additional options to generate
samples. Those methods focus on samples near of the border of the optimal
decision function and will generate samples in the opposite direction of the
nearest neighbors class. Those variants are presented in the figure below.

.. image:: ./auto_examples/over-sampling/images/sphx_glr_plot_comparison_over_sampling_006.png
   :target: ./auto_examples/over-sampling/plot_comparison_over_sampling.html
   :scale: 60
   :align: center


The :class:`BorderlineSMOTE` [HWB2005]_ and :class:`SVMSMOTE` [NCK2009]_ offer
some variant of the SMOTE algorithm::

  >>> from imblearn.over_sampling import BorderlineSMOTE
  >>> X_resampled, y_resampled = BorderlineSMOTE().fit_resample(X, y)
  >>> print(sorted(Counter(y_resampled).items()))
  [(0, 4674), (1, 4674), (2, 4674)]

.. topic:: References

  .. [HWB2005] H. Han, W. Wen-Yuan, M. Bing-Huan, "Borderline-SMOTE: a new
               over-sampling method in imbalanced data sets learning," Advances
               in intelligent computing, 878-887, 2005.

  .. [NCK2009] H. M. Nguyen, E. W. Cooper, K. Kamei, "Borderline over-sampling
               for imbalanced data classification," International Journal of
               Knowledge Engineering and Soft Data Paradigms, 3(1), pp.4-21,
               2009.

Mathematical formulation
========================

Sample generation
-----------------

Both SMOTE and ADASYN use the same algorithm to generate new samples.
Considering a sample :math:`x_i`, a new sample :math:`x_{new}` will be
generated considering its k neareast-neighbors (corresponding to
``k_neighbors``). For instance, the 3 nearest-neighbors are included in the
blue circle as illustrated in the figure below. Then, one of these
nearest-neighbors :math:`x_{zi}` is selected and a sample is generated as
follows:

.. math::

   x_{new} = x_i + \lambda \times (x_{zi} - x_i)

where :math:`\lambda` is a random number in the range :math:`[0, 1]`. This
interpolation will create a sample on the line between :math:`x_{i}` and
:math:`x_{zi}` as illustrated in the image below:

.. image:: ./auto_examples/over-sampling/images/sphx_glr_plot_illustration_generation_sample_001.png
   :target: ./auto_examples/over-sampling/plot_illustration_generation_sample.html
   :scale: 60
   :align: center

Each SMOTE variant and ADASYN differ from each other by selecting the samples
:math:`x_i` ahead of generating the new samples.

The **regular** SMOTE algorithm --- cf. to the :class:`SMOTE` object --- does not
impose any rule and will randomly pick-up all possible :math:`x_i` available.

The **borderline** SMOTE --- cf. to the :class:`BorderlineSMOTE` with the
parameters ``kind='borderline-1'`` and ``kind='borderline-2'`` --- will
classify each sample :math:`x_i` to be (i) noise (i.e. all nearest-neighbors
are from a different class than the one of :math:`x_i`), (ii) in danger
(i.e. at least half of the nearest neighbors are from the same class than
:math:`x_i`, or (iii) safe (i.e. all nearest neighbors are from the same class
than :math:`x_i`). **Borderline-1** and **Borderline-2** SMOTE will use the
samples *in danger* to generate new samples. In **Borderline-1** SMOTE,
:math:`x_{zi}` will belong to the same class than the one of the sample
:math:`x_i`. On the contrary, **Borderline-2** SMOTE will consider
:math:`x_{zi}` which can be from any class.

**SVM** SMOTE --- cf. to :class:`SVMSMOTE` --- uses an SVM classifier to find
support vectors and generate samples considering them. Note that the ``C``
parameter of the SVM classifier allows to select more or less support vectors.

For both borderline and SVM SMOTE, a neighborhood is defined using the
parameter ``m_neighbors`` to decide if a sample is in danger, safe, or noise.

ADASYN is working similarly to the regular SMOTE. However, the number of
samples generated for each :math:`x_i` is proportional to the number of samples
which are not from the same class than :math:`x_i` in a given
neighborhood. Therefore, more samples will be generated in the area that the
nearest neighbor rule is not respected. The parameter ``m_neighbors`` is
equivalent to ``k_neighbors`` in :class:`SMOTE`.

Multi-class management
----------------------

All algorithms can be used with multiple classes as well as binary classes
classification.  :class:`RandomOverSampler` does not require any inter-class
information during the sample generation. Therefore, each targeted class is
resampled independently. In the contrary, both :class:`ADASYN` and
:class:`SMOTE` need information regarding the neighbourhood of each sample used
for sample generation. They are using a one-vs-rest approach by selecting each
targeted class and computing the necessary statistics against the rest of the
data set which are grouped in a single class.
