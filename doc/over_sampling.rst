.. _over-sampling:

=============
Over-sampling
=============

.. currentmodule:: imblearn.over_sampling

As :ref:`discussed earlier <problem_statement>`, the decision function of a
multi-class classifier can favour the majority class, potentially leading to overfitting
(see, for example, a
`Dummy classifier
<https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html>`_).

One approach to address this issue is to generate new samples for the under-represented
classes, a technique known as **over-sampling**.

Please refer to :ref:`sphx_glr_auto_examples_over-sampling_plot_comparison_over_sampling.py`
for details on the visuals included in this document.

.. _random_over_sampler:

Naive Random Over-Sampling
==========================

The most naive strategy is to generate new samples by
**randomly sampling with replacement** from the existing samples. The
:class:`RandomOverSampler` implements this approach::

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

The **augmented data set** `(X_resampled, y_resampled)` should be used
instead of the original data set to train a classifier::

  >>> from sklearn.linear_model import LogisticRegression
  >>> clf = LogisticRegression()
  >>> clf.fit(X_resampled, y_resampled)
  LogisticRegression(...)

In the figure below, we compare the decision functions of a classifier
trained on the augmented dataset with those trained on the original dataset.

.. image:: ./auto_examples/over-sampling/images/sphx_glr_plot_comparison_over_sampling_002.png
   :target: ./auto_examples/over-sampling/plot_comparison_over_sampling.html
   :scale: 60
   :align: center

We observe that the majority class does not dominate the other classes during training.
Consequently, the decision function represents all classes.

In addition, :class:`RandomOverSampler` supports **heterogeneous data**
(e.g., strings, datetime, categorical features, etc.)::

  >>> import numpy as np
  >>> X_hetero = np.array([['xxx', 1, 1.0], ['yyy', 2, 2.0], ['zzz', 3, 3.0]],
  ...                     dtype=object)
  >>> y_hetero = np.array([0, 0, 1])
  >>> X_resampled, y_resampled = ros.fit_resample(X_hetero, y_hetero)
  >>> print(X_resampled)
  [['xxx' 1 1.0]
   ['yyy' 2 2.0]
   ['zzz' 3 3.0]
   ['zzz' 3 3.0]]
  >>> print(y_resampled)
  [0 0 1 1]

It also supports Pandas Dataframes::

  >>> from sklearn.datasets import fetch_openml
  >>> df_adult, y_adult = fetch_openml(
  ...     'adult', version=2, as_frame=True, return_X_y=True)
  >>> df_adult.head()  # doctest: +SKIP
  >>> df_resampled, y_resampled = ros.fit_resample(df_adult, y_adult)
  >>> df_resampled.head()  # doctest: +SKIP

If ordinary repetition is insufficient, the `shrinkage` parameter enables users to perform
a **smoothed bootstrap** (i.e., adding noise to resampled observations). However,
the original data must be numerical.

The `shrinkage` parameter controls the dispersion of the newly generated samples.
We demonstrate that it can be used to produce non-overlapping new samples.
This method of generating a smoothed bootstrap is also known as **Random Over-Sampling Examples
(ROSE)** :cite:`torelli2014rose`.

.. image:: ./auto_examples/over-sampling/images/sphx_glr_plot_comparison_over_sampling_003.png
   :target: ./auto_examples/over-sampling/plot_comparison_over_sampling.html
   :scale: 60
   :align: center

.. _smote_adasyn:

From Random Over-Sampling to SMOTE and ADASYN
=============================================

Apart from the random sampling with replacement, two popular methods
for oversampling minority classes are:

1. **Synthetic Minority Oversampling Technique (SMOTE)** :class:`SMOTE`
:cite:`chawla2002smote`; and

2. **Adaptive Synthetic (ADASYN)** :class:`ADASYN`
:cite:`he2008adasyn`.

These algorithms can be applied in the same way::

  >>> from imblearn.over_sampling import SMOTE, ADASYN
  >>> X_resampled, y_resampled = SMOTE().fit_resample(X, y)
  >>> print(sorted(Counter(y_resampled).items()))
  [(0, 4674), (1, 4674), (2, 4674)]
  >>> clf_smote = LogisticRegression().fit(X_resampled, y_resampled)
  >>> X_resampled, y_resampled = ADASYN().fit_resample(X, y)
  >>> print(sorted(Counter(y_resampled).items()))
  [(0, 4673), (1, 4662), (2, 4674)]
  >>> clf_adasyn = LogisticRegression().fit(X_resampled, y_resampled)

The figure below illustrates the key differences between the various oversampling methods.

.. image:: ./auto_examples/over-sampling/images/sphx_glr_plot_comparison_over_sampling_004.png
   :target: ./auto_examples/over-sampling/plot_comparison_over_sampling.html
   :scale: 60
   :align: center

Ill-Posed Examples
==================

While :class:`RandomOverSampler` over-samples by duplicating samples
from the minority class, :class:`SMOTE` and :class:`ADASYN` generate
new samples through interpolation. However, the approach used to
interpolate or generate these synthetic samples differs.

Specifically, :class:`ADASYN` focuses on generating samples near
the original samples that are misclassified by a k-Nearest Neighbours classifier
(more precisely, a `KDTree
<https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KDTree.html>`_).
In contrast, the basic implementation of :class:`SMOTE` does not distinguish between
easily and difficultly classified samples when using the nearest neighbours rule.
Consequently, the decision functions learned during training will differ between these algorithms.

.. image:: ./auto_examples/over-sampling/images/sphx_glr_plot_comparison_over_sampling_005.png
   :target: ./auto_examples/over-sampling/plot_comparison_over_sampling.html
   :align: center

The specific sampling characteristics of these two algorithms can result in
distinctive behaviours, as demonstrated below.

.. image:: ./auto_examples/over-sampling/images/sphx_glr_plot_comparison_over_sampling_006.png
   :target: ./auto_examples/over-sampling/plot_comparison_over_sampling.html
   :scale: 60
   :align: center

SMOTE Variants
==============

:class:`SMOTE` might connect inliers with outliers; while :class:`ADASYN`
might focus solely on outliers. Both cases can lead to a
sub-optimal decision function. To address this, :class:`SMOTE`
provides three variants for generating samples

1. :class:`BorderlineSMOTE` :cite:`han2005borderline`
2. :class:`SVMSMOTE` :cite:`nguyen2009borderline`
3. :class:`KMeansSMOTE` :cite:`last2017oversampling`

These methods focus on samples near the decision boundary and generate samples
in the opposite direction of the nearest neighbour class.
These variants are illustrated in the figure below.

In particular, the first variant of :class:`BorderlineSMOTE` corresponds to
`kind="borderline-1"`, while the second corresponds to `kind="borderline-2"`.

.. image:: ./auto_examples/over-sampling/images/sphx_glr_plot_comparison_over_sampling_007.png
   :target: ./auto_examples/over-sampling/plot_comparison_over_sampling.html
   :scale: 60
   :align: center

However, none of these SMOTE variants (or, in fact,
any of the methods presented so far, except :class:`RandomOverSampler`) can handle
categorical features. To work with mixed data types (continuous and categorical features),
we introduce the **Synthetic Minority Over-sampling Technique for Nominal and Continuous**
:class:`SMOTENC` :cite:`chawla2002smote`, an extension of the :class:`SMOTE` algorithm
designed to handle categorical features.

We start by creating a dataset that includes both continuous and categorical features::

  >>> # create a synthetic data set with continuous and categorical features
  >>> rng = np.random.RandomState(42)
  >>> n_samples = 50
  >>> X = np.empty((n_samples, 3), dtype=object)
  >>> X[:, 0] = rng.choice(['A', 'B', 'C'], size=n_samples).astype(object)
  >>> X[:, 1] = rng.randn(n_samples)
  >>> X[:, 2] = rng.randint(3, size=n_samples)
  >>> y = np.array([0] * 20 + [1] * 30)
  >>> print(sorted(Counter(y).items()))
  [(0, 20), (1, 30)]

Here, the first and last features are categorical.
This information must be provided to :class:`SMOTENC` via the `categorical_features` parameter
in one of the following ways:

- By relying on `dtype` inference if the columns use the :class:`pandas.CategoricalDtype`.
- By passing the indices of the categorical features when `X` is a Pandas DataFrame.
- By specifying the feature names when `X` is a Pandas DataFrame.
- By providing a Boolean mask identifying the categorical features.

Therefore, the samples generated in the first and last columns belong to the same categories
as the original data, without any additional interpolation::

  >>> from imblearn.over_sampling import SMOTENC
  >>> smote_nc = SMOTENC(categorical_features=[0, 2], random_state=0)
  >>> X_resampled, y_resampled = smote_nc.fit_resample(X, y)
  >>> print(sorted(Counter(y_resampled).items()))
  [(0, 30), (1, 30)]
  >>> print(X_resampled[-5:])
  [['A' 0.19... 2]
   ['B' -0.36... 2]
   ['B' 0.87... 2]
   ['B' 0.37... 2]
   ['B' 0.33... 2]]

However, :class:`SMOTENC` only works when the data is a mixture of continuous and
categorical features. If the data consists only of categorical features,
the **Synthetic Minority Over-sampling Technique for Nominal variant**, :class:`SMOTEN`
:cite:`chawla2002smote` (without the "C"), can be used instead. The algorithm changes in two ways:

- The nearest neighbours search uses the **value difference metric (VDM)**
  :class:`imblearn.metrics.pairwise.ValueDifferenceMetric` instead of Euclidean distance.
- A new sample is generated where each feature value corresponds to the most
  common category among the neighbour samples belonging to the same class.

Let's consider the following example to see how :class:`SMOTEN` handles categorical data::

   >>> import numpy as np
   >>> X = np.array(["green"] * 5 + ["red"] * 10 + ["blue"] * 7,
   ...              dtype=object).reshape(-1, 1)
   >>> y = np.array(["apple"] * 5 + ["not apple"] * 3 + ["apple"] * 7 +
   ...              ["not apple"] * 5 + ["apple"] * 2, dtype=object)

We generate a dataset associating the colours of `apple` and `not apple`.
We strongly associate `green` and `red` with `apple`. The minority class is `not apple`,
so we expect the newly generated data to belong to the category `blue`::

   >>> from imblearn.over_sampling import SMOTEN
   >>> sampler = SMOTEN(random_state=0)
   >>> X_res, y_res = sampler.fit_resample(X, y)
   >>> X_res[y.size:]
   array([['blue'],
           ['blue'],
           ['blue'],
           ['blue'],
           ['blue'],
           ['blue']], dtype=object)
   >>> y_res[y.size:]
   array(['not apple', 'not apple', 'not apple', 'not apple', 'not apple',
          'not apple'], dtype=object)

Sample Generation
=================

Both :class:`SMOTE` and :class:`ADASYN` use the same algorithm to generate new
samples. Given a sample :math:`x_i`, a new sample :math:`x_{new}` will be
generated by considering its :math:`k` nearest-neighbors (corresponding to
``k_neighbors``). For instance, the 3 nearest-neighbors are included in the
blue circle as illustrated in the figure below. Then, one of these
nearest-neighbors :math:`x_{zi}` is selected and a sample is generated as
follows:

Both :class:`SMOTE` and :class:`ADASYN` use the same algorithm to generate new
samples. Given a sample :math:`x_i`, a new sample :math:`x_{new}` is generated
by considering its :math:`k` nearest neighbours (corresponding to the
``k_neighbors`` parameter of :class:`SMOTE`, or ``n_neighbors`` of :class:`ADASYN`).
For example, the three nearest neighbours of :math:`x_i` (including :math:`x_i`
itself) are shown within the blue circle
in the figure below. One of these nearest neighbours, :math:`x_{zi}`, is then selected,
and a new sample is generated as follows:

.. math::

   x_{new} = x_i + \lambda (x_{zi} - x_i)

where :math:`\lambda \in [0,1]` is randomly picked. This
interpolation will create a sample on the line between :math:`x_{i}` and
:math:`x_{zi}` as illustrated in the image below:

.. image:: ./auto_examples/over-sampling/images/sphx_glr_plot_illustration_generation_sample_001.png
   :target: ./auto_examples/over-sampling/plot_illustration_generation_sample.html
   :scale: 60
   :align: center

The sample generation process in :class:`SMOTENC` is slightly different because it
applies a specific approach to categorical features.
Specifically, the category of a newly generated sample is determined by the
most frequent category among its nearest neighbours during the generation process.

.. warning::
   Note that :class:`SMOTENC` is not designed to handle datasets
   consisting solely of categorical features.

The other SMOTE variants and :class:`ADASYN` differ in how they select
the samples :math:x_i before generating new samples:

- :class:`SMOTE` imposes no specific rules and randomly selects
  from all available :math:`x_i`.

- :class:`BorderlineSMOTE` classifies each sample :math:`x_i` into one of
  three categories:

  i. **Noise**: All nearest neighbours belong to a different class than :math:`x_i`.

  ii. **In danger**: At least half of the nearest neighbours belong to the same
      class as :math:`x_i`.

  iii. **Safe**: All nearest neighbours belong to the same class as :math:`x_i`.

  Both ``kind="borderline-1"`` and ``kind="borderline-2"`` use samples
  classified as *in danger* to generate new samples.

  - In ``kind="borderline-1"``, :math:`x_{zi}` is selected from the same class
    as :math:`x_i`.

  - In contrast, ``kind="borderline-2"`` allows :math:`x_{zi}` to be from any class.

- :class:`SVMSMOTE` uses a `SVM classifier
  <https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html>`_
  to identify support vectors and generate samples based on them.
  Note that the ``C`` parameter of the SVM classifier influences the number of support vectors.

For both :class:`BorderlineSMOTE` and :class:`SVMSMOTE`, the neighbourhood used to
determine whether a sample is noise, in danger, or safe is defined by the parameter
``m_neighbors`` rather than ``k_neighbors``.

- :class:`KMeansSMOTE` employs a `k-means clustering method
  <https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html>`_
  before applying :class:`SMOTE`.
  The clustering groups samples together and generates new samples based on the density of each cluster.

- :class:`ADASYN` works similarly to :class:`SMOTE`. However, the number of samples generated for each
  :math:`x_i` is proportional to the number of neighbours that do not belong to the same class as
  :math:`x_i`. Thus, more samples are generated in areas where the *nearest-neighbour rule* is not satisfied.
  The parameter ``m_neighbors`` is equivalent to ``k_neighbors`` in :class:`SMOTE`.

Multi-Class Management
======================

All algorithms can be applied to both binary and multi-class classification.

:class:`RandomOverSampler` does not rely on inter-class information during sample generation,
meaning each target class is resampled independently.

In contrast, both :class:`ADASYN` and :class:`SMOTE` require neighbourhood information
for each sample to generate new ones. These algorithms use a *one-vs-rest* approach,
where each target class is selected, and the necessary statistics are computed against
the rest of the dataset, which is treated as a single class.
