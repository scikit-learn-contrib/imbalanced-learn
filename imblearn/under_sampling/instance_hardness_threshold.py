"""Class to perform under-sampling based on the instance hardness
threshold."""
from __future__ import print_function
from __future__ import division

import numpy as np

from collections import Counter

from sklearn.cross_validation import StratifiedKFold

from ..base import SamplerMixin


ESTIMATOR_KIND = ('knn', 'decision-tree', 'random-forest', 'adaboost',
                  'gradient-boosting', 'linear-svm')


class InstanceHardnessThreshold(SamplerMixin):
    """Class to perform under-sampling based on the instance hardness
    threshold.

    Parameters
    ----------
    estimator : str, optional (default='linear-svm')
        Classifier to be used in to estimate instance hardness of the samples.
        The choices are the following: 'knn',
        'decision-tree', 'random-forest', 'adaboost', 'gradient-boosting'
        and 'linear-svm'.

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

    n_jobs : int, optional (default=-1)
        The number of threads to open if possible.

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

    References
    ----------
    .. [1] D. Smith, Michael R., Tony Martinez, and Christophe Giraud-Carrier.
       "An instance level analysis of data complexity." Machine learning
       95.2 (2014): 225-256.

    """

    def __init__(self, estimator='linear-svm', ratio='auto',
                 return_indices=False, random_state=None, cv=5, n_jobs=-1,
                 **kwargs):
        super(InstanceHardnessThreshold, self).__init__(ratio=ratio)
        self.estimator = estimator
        self.return_indices = return_indices
        self.random_state = random_state
        self.kwargs = kwargs
        self.cv = cv
        self.n_jobs = n_jobs

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

        if self.estimator not in ESTIMATOR_KIND:
            raise NotImplementedError

        # Select the appropriate classifier
        if self.estimator == 'knn':
            from sklearn.neighbors import KNeighborsClassifier
            estimator = KNeighborsClassifier(
                **self.kwargs)
        elif self.estimator == 'decision-tree':
            from sklearn.tree import DecisionTreeClassifier
            estimator = DecisionTreeClassifier(
                random_state=self.random_state,
                **self.kwargs)
        elif self.estimator == 'random-forest':
            from sklearn.ensemble import RandomForestClassifier
            estimator = RandomForestClassifier(
                random_state=self.random_state,
                **self.kwargs)
        elif self.estimator == 'adaboost':
            from sklearn.ensemble import AdaBoostClassifier
            estimator = AdaBoostClassifier(
                random_state=self.random_state,
                **self.kwargs)
        elif self.estimator == 'gradient-boosting':
            from sklearn.ensemble import GradientBoostingClassifier
            estimator = GradientBoostingClassifier(
                random_state=self.random_state,
                **self.kwargs)
        elif self.estimator == 'linear-svm':
            from sklearn.svm import SVC
            estimator = SVC(probability=True,
                            random_state=self.random_state, **self.kwargs)
        else:
            raise NotImplementedError

        # Create the different folds
        skf = StratifiedKFold(y, n_folds=self.cv, shuffle=False,
                              random_state=self.random_state)

        probabilities = np.zeros(y.shape[0], dtype=float)

        for train_index, test_index in skf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            estimator.fit(X_train, y_train)

            probs = estimator.predict_proba(X_test)
            classes = estimator.classes_
            probabilities[test_index] = [
                probs[l, np.where(classes == c)[0][0]]
                for l, c in enumerate(y_test)]

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

        self.logger.info('Under-sampling performed: %s', Counter(
            y_resampled))

        # If we need to offer support for the indices
        if self.return_indices:
            idx_under = np.nonzero(mask)[0]
            return X_resampled, y_resampled, idx_under
        else:
            return X_resampled, y_resampled
