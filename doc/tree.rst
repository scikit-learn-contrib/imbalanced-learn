.. _tree-split:

==============
Tree-split
==============

.. currentmodule:: imblearn.tree

.. _cluster_centroids:


Hellinger Distance split
====================

Hellinger Distance is used to quantify the similarity between two probability distributions.
When used as split criterion in Decision Tree Classifier it makes it skew insensitive and helps tackle the imbalance problem.

  >>> import numpy as np
  >>> from sklearn.ensemble import RandomForestClassifier
  >>> from imblearn.tree.criterion import HellingerDistanceCriterion

  >>> hdc = HellingerDistanceCriterion(1, np.array([2],dtype='int64'))
  >>> clf = RandomForestClassifier(criterion=hdc)

:class:`HellingerDistanceCriterion` offers a Cython implementation of Hellinger Distance as a criterion for decision tree split compatible with sklearn tree based classification models.
