.. _over-sampling:

=============
Over-sampling
=============

.. currentmodule:: imblearn.over_sampling

A practical guide
=================

The learning phase and the subsequent prediction of machine learning algorithms
can be affected by the problem of imbalanced data set. The balancing issue
corresponds to the difference of the number of samples in the different
classes. We illustrate the effect of training a linear SVM classifier with
different level of class balancing.

.. image:: ./modules/balancing_problem/linear_svc_imbalanced_issue.png
   :scale: 80
   :align: center

As expected, the decision function of the linear SVM is highly impacted. With a
greater imbalanced ratio, the decision function favor the class with the larger
number of samples, usually referred as the majority class.

One way to fight this issue is to generate new samples in the classes which are
under-represented. The most naive strategy is to generate new samples by
randomly sampling with replacement the current available samples. The
:class:`RandomOverSampler` offers such scheme::

   >>> from sklearn.datasets import make_classification
   >>> X, y = make_classification(n_samples=5000, n_features=2, n_informative=2,
                                  n_redundant=0, n_repeated=0, n_classes=3,
                                  n_clusters_per_class=1,
                                  weights=[0.01, 0.05, 0.94],
                                  class_sep=0.8, random_state=0)
   >>> from imblearn.over_sampling import RandomOverSampler
   >>> rus = RandomOverSampler(random_state=0)
   >>> X_resampled, y_resampled = rus.fit_sample(X, y)
   >>> from collections import Counter
   >>> print(Counter(y_res))

The augmented data set should be used instead of the original data set to train
a classifier::

  >>> from sklearn.svm import LinearSVC
  >>> clf = LinearSVC()
  >>> clf.fit(X_resampled, y_resampled)

In the figure below, we compare the decision functions of a classifier trained
using the over-sampled data set and the original data set.

.. image:: ./modules/over_sampling/random_over_sampler.png
   :scale: 80
   :align: center

As a result, the majority class does not take over the other classes during the
training process. Consequently, all classes are represented by the decision
function.

Mathematical formulation
========================

SMOTE
-----

Regular SMOTE
~~~~~~~~~~~~~

Borderline SMOTE
~~~~~~~~~~~~~~~~

SVM SMOTE
~~~~~~~~~

ADASYN
------
