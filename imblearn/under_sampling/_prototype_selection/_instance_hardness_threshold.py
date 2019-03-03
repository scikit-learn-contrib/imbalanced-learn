"""Class to perform under-sampling based on the instance hardness
threshold."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Dayvid Oliveira
#          Christos Aridas
# License: MIT

from __future__ import division

from collections import Counter

import numpy as np

from sklearn.base import ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import safe_indexing

from ..base import BaseUnderSampler
from ...utils import Substitution
from ...utils.deprecation import deprecate_parameter
from ...utils._docstring import _random_state_docstring


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring)
class InstanceHardnessThreshold(BaseUnderSampler):
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

    {sampling_strategy}

    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly
        selected.

        .. deprecated:: 0.4
           ``return_indices`` is deprecated. Use the attribute
           ``sample_indices_`` instead.

    {random_state}

    cv : int, optional (default=5)
        Number of folds to be used when estimating samples' instance hardness.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    ratio : str, dict, or callable
        .. deprecated:: 0.4
           Use the parameter ``sampling_strategy`` instead. It will be removed
           in 0.6.

    Attributes
    ----------
    sample_indices_ : ndarray, shape (n_new_samples)
        Indices of the samples selected.

        .. versionadded:: 0.4
           ``sample_indices_`` used instead of ``return_indices=True``.

    Notes
    -----
    The method is based on [1]_.

    Supports multi-class resampling. A one-vs.-rest scheme is used when
    sampling a class as proposed in [1]_.

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
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> iht = InstanceHardnessThreshold(random_state=42)
    >>> X_res, y_res = iht.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{1: 574, 0: 100}})

    """

    def __init__(self,
                 estimator=None,
                 sampling_strategy='auto',
                 return_indices=False,
                 random_state=None,
                 cv=5,
                 n_jobs=1,
                 ratio=None):
        super(InstanceHardnessThreshold, self).__init__(
            sampling_strategy=sampling_strategy, ratio=ratio)
        self.random_state = random_state
        self.estimator = estimator
        self.return_indices = return_indices
        self.cv = cv
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Private function to create the classifier"""

        if (self.estimator is not None and
                isinstance(self.estimator, ClassifierMixin) and
                hasattr(self.estimator, 'predict_proba')):
            self.estimator_ = clone(self.estimator)
        elif self.estimator is None:
            self.estimator_ = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state,
                n_jobs=self.n_jobs)
        else:
            raise ValueError('Invalid parameter `estimator`. Got {}.'.format(
                type(self.estimator)))

    def _fit_resample(self, X, y):
        if self.return_indices:
            deprecate_parameter(self, '0.4', 'return_indices',
                                'sample_indices_')
        self._validate_estimator()

        target_stats = Counter(y)
        skf = StratifiedKFold(
            n_splits=self.cv, shuffle=False,
            random_state=self.random_state).split(X, y)
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
            if target_class in self.sampling_strategy_.keys():
                n_samples = self.sampling_strategy_[target_class]
                threshold = np.percentile(
                    probabilities[y == target_class],
                    (1. - (n_samples / target_stats[target_class])) * 100.)
                index_target_class = np.flatnonzero(
                    probabilities[y == target_class] >= threshold)
            else:
                index_target_class = slice(None)

            idx_under = np.concatenate(
                (idx_under,
                 np.flatnonzero(y == target_class)[index_target_class]),
                axis=0)

        self.sample_indices_ = idx_under

        if self.return_indices:
            return (safe_indexing(X, idx_under), safe_indexing(y, idx_under),
                    idx_under)
        return safe_indexing(X, idx_under), safe_indexing(y, idx_under)
