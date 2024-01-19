.. _ensemble:

====================
Ensemble of samplers
====================

.. currentmodule:: imblearn.ensemble

.. _ensemble_meta_estimators:

Classifier including inner balancing samplers
=============================================

.. _bagging:

Bagging classifier
------------------

In ensemble classifiers, bagging methods build several estimators on different
randomly selected subset of data. In scikit-learn, this classifier is named
:class:`~sklearn.ensemble.BaggingClassifier`. However, this classifier does not
allow to balance each subset of data. Therefore, when training on imbalanced
data set, this classifier will favor the majority classes::

  >>> from sklearn.datasets import make_classification
  >>> X, y = make_classification(n_samples=10000, n_features=2, n_informative=2,
  ...                            n_redundant=0, n_repeated=0, n_classes=3,
  ...                            n_clusters_per_class=1,
  ...                            weights=[0.01, 0.05, 0.94], class_sep=0.8,
  ...                            random_state=0)
  >>> from sklearn.model_selection import train_test_split
  >>> from sklearn.metrics import balanced_accuracy_score
  >>> from sklearn.ensemble import BaggingClassifier
  >>> from sklearn.tree import DecisionTreeClassifier
  >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
  >>> bc = BaggingClassifier(DecisionTreeClassifier(), random_state=0)
  >>> bc.fit(X_train, y_train) #doctest:
  BaggingClassifier(...)
  >>> y_pred = bc.predict(X_test)
  >>> balanced_accuracy_score(y_test, y_pred)
  0.77...

In :class:`BalancedBaggingClassifier`, each bootstrap sample will be further
resampled to achieve the `sampling_strategy` desired. Therefore,
:class:`BalancedBaggingClassifier` takes the same parameters as the
scikit-learn :class:`~sklearn.ensemble.BaggingClassifier`. In addition, the
sampling is controlled by the parameter `sampler` or the two parameters
`sampling_strategy` and `replacement`, if one wants to use the
:class:`~imblearn.under_sampling.RandomUnderSampler`::

  >>> from imblearn.ensemble import BalancedBaggingClassifier
  >>> bbc = BalancedBaggingClassifier(DecisionTreeClassifier(),
  ...                                 sampling_strategy='auto',
  ...                                 replacement=False,
  ...                                 random_state=0)
  >>> bbc.fit(X_train, y_train)
  BalancedBaggingClassifier(...)
  >>> y_pred = bbc.predict(X_test)
  >>> balanced_accuracy_score(y_test, y_pred)
  0.8...

Changing the `sampler` will give rise to different known implementation
:cite:`maclin1997empirical`, :cite:`hido2009roughly`,
:cite:`wang2009diversity`. You can refer to the following example shows in
practice these different methods:
:ref:`sphx_glr_auto_examples_ensemble_plot_bagging_classifier.py`

.. _forest:

Forest of randomized trees
--------------------------

:class:`BalancedRandomForestClassifier` is another ensemble method in which
each tree of the forest will be provided a balanced bootstrap sample
:cite:`chen2004using`. This class provides all functionality of the
:class:`~sklearn.ensemble.RandomForestClassifier`::

  >>> from imblearn.ensemble import BalancedRandomForestClassifier
  >>> brf = BalancedRandomForestClassifier(
  ...     n_estimators=100, random_state=0, sampling_strategy="all", replacement=True,
  ...     bootstrap=False,
  ... )
  >>> brf.fit(X_train, y_train)
  BalancedRandomForestClassifier(...)
  >>> y_pred = brf.predict(X_test)
  >>> balanced_accuracy_score(y_test, y_pred)
  0.8...

.. _boosting:

Boosting
--------

Several methods taking advantage of boosting have been designed.

:class:`RUSBoostClassifier` randomly under-sample the dataset before to perform
a boosting iteration :cite:`seiffert2009rusboost`::

  >>> from imblearn.ensemble import RUSBoostClassifier
  >>> rusboost = RUSBoostClassifier(n_estimators=200, algorithm='SAMME.R',
  ...                               random_state=0)
  >>> rusboost.fit(X_train, y_train)
  RUSBoostClassifier(...)
  >>> y_pred = rusboost.predict(X_test)
  >>> balanced_accuracy_score(y_test, y_pred)
  0...

A specific method which uses :class:`~sklearn.ensemble.AdaBoostClassifier` as
learners in the bagging classifier is called "EasyEnsemble". The
:class:`EasyEnsembleClassifier` allows to bag AdaBoost learners which are
trained on balanced bootstrap samples :cite:`liu2008exploratory`. Similarly to
the :class:`BalancedBaggingClassifier` API, one can construct the ensemble as::

  >>> from imblearn.ensemble import EasyEnsembleClassifier
  >>> eec = EasyEnsembleClassifier(random_state=0)
  >>> eec.fit(X_train, y_train)
  EasyEnsembleClassifier(...)
  >>> y_pred = eec.predict(X_test)
  >>> balanced_accuracy_score(y_test, y_pred)
  0.6...

.. topic:: Examples

  * :ref:`sphx_glr_auto_examples_ensemble_plot_comparison_ensemble_classifier.py`
