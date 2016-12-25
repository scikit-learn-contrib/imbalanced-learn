"""Class to perform under-sampling based on the instance hardness
threshold."""
from __future__ import division, print_function

import warnings
from collections import Counter

import numpy as np
from six import string_types
import sklearn
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

from ..base import BaseBinarySampler


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


class InstanceHardnessThreshold(BaseBinarySampler):
    """Class to perform under-sampling based on the instance hardness
    threshold.

    Parameters
    ----------
    estimator : object, optional (default=RandomForestClassifier())
        Classifier to be used to estimate instance hardness of the samples.
        By default a RandomForestClassifer will be used.
        If str, the choices using a string are the following: 'knn',
        'decision-tree', 'random-forest', 'adaboost', 'gradient-boosting'
        and 'linear-svm'.
        If object, an estimator inherited from `sklearn.base.ClassifierMixin`
        and having an attribute `predict_proba`.

        NOTE: `estimator` as a string object is deprecated from 0.2 and will be
        replaced in 0.4. Use `ClassifierMixin` object instead.

    ratio : str or float, optional (default='auto')
        If 'auto', the ratio will be defined automatically to balance
        the dataset. Otherwise, the ratio is defined as the number
        of samples in the minority class over the the number of samples
        in the majority class.

    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly
        selected from the majority class.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    cv : int, optional (default=5)
        Number of folds to be used when estimating samples' instance hardness.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    **kwargs:
        Option for the different classifier.

        NOTE: `**kwargs` has been deprecated from 0.2 and will be replaced in
        0.4. Use `ClassifierMixin` object instead to pass parameter associated
        to an estimator.

    Attributes
    ----------
    min_c_ : str or int
        The identifier of the minority class.

    max_c_ : str or int
        The identifier of the majority class.

    stats_c_ : dict of str/int : int
        A dictionary in which the number of occurences of each class is
        reported.

    cv : int, optional (default=5)
        Number of folds used when estimating samples' instance hardness.

    X_shape_ : tuple of int
        Shape of the data `X` during fitting.

    Notes
    -----
    The method is based on [1]_.

    This class does not support multi-class.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import \
    RepeatedEditedNearestNeighbours # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> renn = RepeatedEditedNearestNeighbours(random_state=42)
    >>> X_res, y_res = renn.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({1: 883, 0: 100})

    References
    ----------
    .. [1] D. Smith, Michael R., Tony Martinez, and Christophe Giraud-Carrier.
       "An instance level analysis of data complexity." Machine learning
       95.2 (2014): 225-256.

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
            raise ValueError('Invalid parameter `estimator`')

    def fit(self, X, y):
        """Find the classes statistics before to perform sampling.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        self : object,
            Return self.

        """

        super(InstanceHardnessThreshold, self).fit(X, y)

        self._validate_estimator()

        return self

    def _sample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : ndarray, shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new)
            The corresponding label of `X_resampled`

        idx_under : ndarray, shape (n_samples, )
            If `return_indices` is `True`, a boolean array will be returned
            containing the which samples have been selected.

        """

        # Create the different folds
        skf = _get_cv_splits(X, y, self.cv, self.random_state)

        probabilities = np.zeros(y.shape[0], dtype=float)

        for train_index, test_index in skf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.estimator_.fit(X_train, y_train)

            probs = self.estimator_.predict_proba(X_test)
            classes = self.estimator_.classes_
            probabilities[test_index] = [
                probs[l, np.where(classes == c)[0][0]]
                for l, c in enumerate(y_test)
            ]

        # Compute the number of cluster needed
        if self.ratio == 'auto':
            num_samples = self.stats_c_[self.min_c_]
        else:
            num_samples = int(self.stats_c_[self.min_c_] / self.ratio)

        # Find the percentile corresponding to the top num_samples
        threshold = np.percentile(
            probabilities[y != self.min_c_],
            (1. - (num_samples / self.stats_c_[self.maj_c_])) * 100.)

        mask = np.logical_or(probabilities >= threshold, y == self.min_c_)

        # Sample the data
        X_resampled = X[mask]
        y_resampled = y[mask]

        self.logger.info('Under-sampling performed: %s', Counter(y_resampled))

        # If we need to offer support for the indices
        if self.return_indices:
            idx_under = np.flatnonzero(mask)
            return X_resampled, y_resampled, idx_under
        else:
            return X_resampled, y_resampled
