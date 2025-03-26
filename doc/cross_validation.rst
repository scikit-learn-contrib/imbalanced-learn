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

We will discuss instance hardness in this document and explain how to use the
`InstanceHardnessCV` splitter.

Instance hardness and average precision
=======================================

Let’s start by creating a dataset to work with. We create a dataset with 5% class
imbalance using scikit-learn’s `make_blobs` function.

  >>> import numpy as np
  >>> from matplotlib import pyplot as plt
  >>> from sklearn.datasets import make_blobs
  >>> from imblearn.datasets import make_imbalance
  >>> random_state = 10
  >>> X, y = make_blobs(n_samples=[950, 50], centers=((-3, 0), (3, 0)),
  ...                   random_state=random_state)
  >>> plt.scatter(X[:, 0], X[:, 1], c=y)
  >>> plt.show()

.. image:: ./auto_examples/cross_validation/images/sphx_glr_plot_instance_hardness_cv_001.png
   :target: ./auto_examples/cross_validation/plot_instance_hardness_cv.html
   :align: center

Now we add some samples with large instance hardness

  >>> X_hard, y_hard = make_blobs(n_samples=10, centers=((3, 0), (-3, 0)),
  ...                             cluster_std=1,
  ...                             random_state=random_state)
  >>> X = np.vstack((X, X_hard))
  >>> y = np.hstack((y, y_hard))
  >>> plt.scatter(X[:, 0], X[:, 1], c=y)
  >>> plt.show()

.. image:: ./auto_examples/cross_validation/images/sphx_glr_plot_instance_hardness_cv_002.png
   :target: ./auto_examples/cross_validation/plot_instance_hardness_cv.html
   :align: center

Then we take a `LogisticRegressionClassifier` and assess the cross validation
performance using a `StratifiedKFold` cv splitter and the `cross_validate`
function.

  >>> from sklearn.ensemble import LogisticRegressionClassifier
  >>> clf = LogisticRegressionClassifier(random_state=random_state)
  >>> skf_cv = StratifiedKFold(n_splits=5, shuffle=True,
  ...                           random_state=random_state)
  >>> skf_result = cross_validate(clf, X, y, cv=skf_cv, scoring="average_precision")

Now, we do the same using an `InstanceHardnessCV` splitter. We use provide our
classifier to the splitter to calculate instance hardness and distribute samples
with large instance hardness equally over the folds.

  >>> ih_cv = InstanceHardnessCV(n_splits=5, estimator=clf,
  ...                               random_state=random_state)
  >>> ih_result = cross_validate(clf, X, y, cv=ih_cv, scoring="average_precision")

When we plot the test scores for both cv splitters, we see that the variance using
the `InstanceHardnessCV` splitter is lower than for the `StratifiedKFold` splitter.

  >>> plt.boxplot([skf_result['test_score'], ih_result['test_score']],
  ...               tick_labels=["StratifiedKFold", "InstanceHardnessCV"],
  ...               vert=False)
  >>> plt.tight_layout()

.. image:: ./auto_examples/cross_validation/images/sphx_glr_plot_instance_hardness_cv_003.png
   :target: ./auto_examples/cross_validation/plot_instance_hardness_cv.html
   :align: center