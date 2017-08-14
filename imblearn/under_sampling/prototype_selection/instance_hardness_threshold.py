"""Class to perform under-sampling based on the instance hardness
threshold."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Dayvid Oliveira
#          Christos Aridas
# License: MIT

from __future__ import division

import warnings
from collections import Counter

import numpy as np
import sklearn
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.six import string_types
from sklearn.utils import safe_indexing

from ..base import BaseCleaningSampler


def _get_cv_splits(X, y, cv, random_state):
    if hasattr(sklearn, 'model_selection'):
        from sklearn.model_selection import StratifiedKFold
        cv_iterator = StratifiedKFold(
            n_splits=cv, shuffle=False, random_state=random_state).split(X, y)
    else:
        from sklearn.cross_validation import StratifiedKFold
        cv_iterator = StratifiedKFold(
            y, n_folds=cv, shuffle=False, random_state=random_state)

    return cv_iterator


class InstanceHardnessThreshold(BaseCleaningSampler):
    """Class to perform under-sampling based on the instance hardness
    threshold.

    Read more in the :ref:`User Guide <instance_hardness_threshold>`.

    Parameters
    ----------
    estimator : object, optional (default=RandomForestClassifier())
        Classifier to be used to estimate instance hardness of the samples.  By
        default a :class:`sklearn.ensemble.RandomForestClassifer` will be used.
        If ``str``, the choices using a string are the following: ``'knn'``,
        ``'decision-tree'``, ``'random-forest'``, ``'adaboost'``,
        ``'gradient-boosting'`` and ``'linear-svm'``.  If object, an estimator
        inherited from :class:`sklearn.base.ClassifierMixin` and having an
        attribute :func:`predict_proba`.

        .. deprecated:: 0.2
           ``estimator`` as a string object is deprecated from 0.2 and will be
           replaced in 0.4. Use :class:`sklearn.base.ClassifierMixin` object
           instead.

    ratio : str, dict, or callable, optional (default='auto')
        Ratio to use for resampling the data set.

        - If ``str``, has to be one of: (i) ``'minority'``: resample the
          minority class; (ii) ``'majority'``: resample the majority class,
          (iii) ``'not minority'``: resample all classes apart of the minority
          class, (iv) ``'all'``: resample all classes, and (v) ``'auto'``:
          correspond to ``'all'`` with for over-sampling methods and ``'not
          minority'`` for under-sampling methods. The classes targeted will be
          over-sampled or under-sampled to achieve an equal number of sample
          with the majority or minority class.
        - If ``dict``, the keys correspond to the targeted classes. The values
          correspond to the desired number of samples.
        - If callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples.

        .. warning::
           This algorithm is a cleaning under-sampling method. When providing a
           ``dict``, only the targeted classes will be used; the number of
           samples will be discarded.

    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly
        selected from the majority class.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number
        generator; If ``RandomState`` instance, random_state is the random
        number generator; If ``None``, the random number generator is the
        ``RandomState`` instance used by ``np.random``.

    cv : int, optional (default=5)
        Number of folds to be used when estimating samples' instance hardness.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    **kwargs:
        Option for the different classifier.

        .. deprecated:: 0.2
           ``**kwargs`` has been deprecated from 0.2 and will be replaced in
           0.4. Use :class:`sklearn.base.ClassifierMixin` object instead to
           pass parameter associated to an estimator.

    Notes
    -----
    The method is based on [1]_.

    Supports mutli-class resampling. A one-vs.-rest scheme is used when
    sampling a class as proposed in [1]_.

    See
    :ref:`sphx_glr_auto_examples_under-sampling_plot_instance_hardness_threshold.py`.

    References
    ----------
    .. [1] D. Smith, Michael R., Tony Martinez, and Christophe Giraud-Carrier.
       "An instance level analysis of data complexity." Machine learning
       95.2 (2014): 225-256.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import InstanceHardnessThreshold
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> iht = InstanceHardnessThreshold(random_state=42)
    >>> X_res, y_res = iht.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({1: 840, 0: 100})

    """

    def __init__(self,
                 estimator=None,
                 ratio='auto',
                 return_indices=False,
                 random_state=None,
                 cv=5,
                 n_jobs=1,
                 **kwargs):
        super(InstanceHardnessThreshold, self).__init__(
            ratio=ratio, random_state=random_state)
        self.estimator = estimator
        self.return_indices = return_indices
        self.cv = cv
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def _validate_estimator(self):
        """Private function to create the classifier"""

        if (self.estimator is not None and
                isinstance(self.estimator, ClassifierMixin) and
                hasattr(self.estimator, 'predict_proba')):
            self.estimator_ = self.estimator
        elif self.estimator is None:
            self.estimator_ = RandomForestClassifier(
                random_state=self.random_state, n_jobs=self.n_jobs)
        # To be removed in 0.4
        elif (self.estimator is not None and
              isinstance(self.estimator, string_types)):
            # Select the appropriate classifier
            warnings.warn('`estimator` will be replaced in version'
                          ' 0.4. Use a classifier object instead of a string.',
                          DeprecationWarning)
            if self.estimator == 'knn':
                from sklearn.neighbors import KNeighborsClassifier
                self.estimator_ = KNeighborsClassifier(**self.kwargs)
            elif self.estimator == 'decision-tree':
                from sklearn.tree import DecisionTreeClassifier
                self.estimator_ = DecisionTreeClassifier(
                    random_state=self.random_state, **self.kwargs)
            elif self.estimator == 'random-forest':
                self.estimator_ = RandomForestClassifier(
                    random_state=self.random_state, **self.kwargs)
            elif self.estimator == 'adaboost':
                from sklearn.ensemble import AdaBoostClassifier
                self.estimator_ = AdaBoostClassifier(
                    random_state=self.random_state, **self.kwargs)
            elif self.estimator == 'gradient-boosting':
                from sklearn.ensemble import GradientBoostingClassifier
                self.estimator_ = GradientBoostingClassifier(
                    random_state=self.random_state, **self.kwargs)
            elif self.estimator == 'linear-svm':
                from sklearn.svm import SVC
                self.estimator_ = SVC(probability=True,
                                      random_state=self.random_state,
                                      kernel='linear',
                                      **self.kwargs)
            else:
                raise NotImplementedError
        else:
            raise ValueError('Invalid parameter `estimator`. Got {}.'.format(
                type(self.estimator)))

    def _sample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like, shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {ndarray, sparse matrix}, shape \
(n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new,)
            The corresponding label of `X_resampled`

        idx_under : ndarray, shape (n_samples, )
            If `return_indices` is `True`, a boolean array will be returned
            containing the which samples have been selected.

        """
        self._validate_estimator()

        target_stats = Counter(y)
        skf = _get_cv_splits(X, y, self.cv, self.random_state)
        probabilities = np.zeros(y.shape[0], dtype=float)

        for train_index, test_index in skf:
            X_train = safe_indexing(X, train_index)
            X_test = safe_indexing(X, test_index)
            y_train = safe_indexing(y, train_index)
            y_test = safe_indexing(y, test_index)

            self.estimator_.fit(X_train, y_train)

            probs = self.estimator_.predict_proba(X_test)
            classes = self.estimator_.classes_
            probabilities[test_index] = [
                probs[l, np.where(classes == c)[0][0]]
                for l, c in enumerate(y_test)
            ]

        idx_under = np.empty((0, ), dtype=int)

        for target_class in np.unique(y):
            if target_class in self.ratio_.keys():
                n_samples = self.ratio_[target_class]
                threshold = np.percentile(
                    probabilities[y == target_class],
                    (1. - (n_samples / target_stats[target_class])) * 100.)
                index_target_class = np.flatnonzero(
                    probabilities[y == target_class] >= threshold)
            else:
                index_target_class = slice(None)

            idx_under = np.concatenate(
                (idx_under, np.flatnonzero(y == target_class)[
                    index_target_class]), axis=0)

        if self.return_indices:
            return (safe_indexing(X, idx_under), safe_indexing(y, idx_under),
                    idx_under)
        else:
            return safe_indexing(X, idx_under), safe_indexing(y, idx_under)
