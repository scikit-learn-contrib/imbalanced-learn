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
``BaggingClassifier``. However, this classifier does not allow to balance each
subset of data. Therefore, when training on imbalanced data set, this
classifier will favor the majority classes::

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
  >>> bc = BaggingClassifier(base_estimator=DecisionTreeClassifier(),
  ...                        random_state=0)
  >>> bc.fit(X_train, y_train) #doctest: +ELLIPSIS
  BaggingClassifier(...)
  >>> y_pred = bc.predict(X_test)
  >>> balanced_accuracy_score(y_test, y_pred)  # doctest: +ELLIPSIS
  0.77...

:class:`BalancedBaggingClassifier` allows to resample each subset of data
before to train each estimator of the ensemble. In short, it combines the
output of an :class:`EasyEnsemble` sampler with an ensemble of classifiers
(i.e. ``BaggingClassifier``). Therefore, :class:`BalancedBaggingClassifier`
takes the same parameters than the scikit-learn
``BaggingClassifier``. Additionally, there is two additional parameters,
``sampling_strategy`` and ``replacement`` to control the behaviour of the
random under-sampler::

  >>> from imblearn.ensemble import BalancedBaggingClassifier
  >>> bbc = BalancedBaggingClassifier(base_estimator=DecisionTreeClassifier(),
  ...                                 sampling_strategy='auto',
  ...                                 replacement=False,
  ...                                 random_state=0)
  >>> bbc.fit(X_train, y_train) # doctest: +ELLIPSIS
  BalancedBaggingClassifier(...)
  >>> y_pred = bbc.predict(X_test)
  >>> balanced_accuracy_score(y_test, y_pred)  # doctest: +ELLIPSIS
  0.80...

.. _forest:

Forest of randomized trees
--------------------------

:class:`BalancedRandomForestClassifier` is another ensemble method in which
each tre1vided a balanced boostrap sample [1CLB2004]_. This class
provides all functionality of the
:class:`sklearn.ensemble.RandomForestClassifier` and notably the
`feature_importances_` attributes::

  >>> from imblearn.ensemble import BalancedRandomForestClassifier
  >>> brf = BalancedRandomForestClassifier(n_estimators=100, random_state=0)
  >>> brf.fit(X_train, y_train) # doctest: +ELLIPSIS
  BalancedRandomForestClassifier(...)
  >>> y_pred = brf.predict(X_test)
  >>> balanced_accuracy_score(y_test, y_pred)  # doctest: +ELLIPSIS
  0.80...
  >>> brf.feature_importances_  # doctest: +ELLIPSIS
  array([ 0.57...,  0.42...])

.. _boosting:

Boosting
--------

Several methods taking advantage of boosting have been designed.

:class:`RUSBoostClassifier` randomly under-sample the dataset before to perform
a boosting iteration [SKHN2010]_::

  >>> from imblearn.ensemble import RUSBoostClassifier
  >>> rusboost = RUSBoostClassifier(random_state=0)
  >>> rusboost.fit(X_train, y_train)  # doctest: +ELLIPSIS
  RUSBoostClassifier(...)
  >>> y_pred = rusboost.predict(X_test)
  >>> balanced_accuracy_score(y_test, y_pred)  # doctest: +ELLIPSIS
  0.74...

A specific method which uses ``AdaBoost`` as learners in the bagging classifier
is called EasyEnsemble. The :class:`EasyEnsembleClassifier` allows to bag
AdaBoost learners which are trained on balanced bootstrap samples [LWZ2009]_.
Similarly to the :class:`BalancedBaggingClassifier` API, one can construct the
ensemble as::

  >>> from imblearn.ensemble import EasyEnsembleClassifier
  >>> eec = EasyEnsembleClassifier(random_state=0)
  >>> eec.fit(X_train, y_train) # doctest: +ELLIPSIS
  EasyEnsembleClassifier(...)
  >>> y_pred = eec.predict(X_test)
  >>> balanced_accuracy_score(y_test, y_pred)  # doctest: +ELLIPSIS
  0.62...

.. topic:: Examples

  * :ref:`sphx_glr_auto_examples_ensemble_plot_comparison_ensemble_classifier.py`

.. topic:: References

  .. [1CLB2004] Chen, Chao, Andy Liaw, and Leo Breiman. "Using random forest to
                learn imbalanced data." University of California, Berkeley 110
                (2004): 1-12.

  .. [LWZ2009] X. Y. Liu, J. Wu and Z. H. Zhou, "Exploratory Undersampling for
               Class-Imbalance Learning," in IEEE Transactions on Systems, Man,
               and Cybernetics, Part B (Cybernetics), vol. 39, no. 2, pp.
               539-550, April 2009.

  .. [SKHN2010] Seiffert, C., Khoshgoftaar, T. M., Van Hulse, J., &
                Napolitano, A. "RUSBoost: A hybrid approach to alleviating
                class imbalance." IEEE Transactions on Systems, Man, and
                Cybernetics-Part A: Systems and Humans 40.1 (2010): 185-197.