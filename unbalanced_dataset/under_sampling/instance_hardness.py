"""Class to perform under-sampling based on the instance hardness
threshold."""
from __future__ import print_function
from __future__ import division

import numpy as np

from collections import Counter

from scipy.stats import mode

from sklearn.utils import check_X_y
from sklearn.neighbors import NearestNeighbors
from sklearn.cross_validation import StratifiedKFold

from .under_sampler import UnderSampler


class InstanceHardnessThreshold(UnderSampler):
    """Class to perform under-sampling based on the instance hardness 
    threshold.

    Parameters
    ----------
    estimator : sklearn classifier
        Classifier to be used in to estimate instance hardness of the samples.

    threshold : float, optional (default=0.3)
        Threshold to be used when excluding samples (0.01 to 0.99).

    mode: str, optional (default='maj')
        - If 'maj', only samples of the majority class are excluded.
        - If 'all', samples of all classes are excluded.

    cv : int, optional (default=5)
        Number of folds to be used when estimating samples' instance hardness.

    return_indices : bool, optional (default=False)
        Either to return or not the indices which will be selected from
        the majority class.

    random_state : int or None, optional (default=None)
        Seed for random number generation.

    verbose : bool, optional (default=True)
        Boolean to either or not print information about the processing

    n_jobs : int, optional (default=-1)
        The number of thread to open when it is possible.


    Attributes
    ----------
    ratio_ : str or float, optional (default='auto')
        If 'auto', the ratio will be defined automatically to balanced
        the dataset. Otherwise, the ratio will corresponds to the number
        of samples in the minority class over the the number of samples
        in the majority class.

    rs_ : int or None, optional (default=None)
        Seed for random number generation.

    min_c_ : str or int
        The identifier of the minority class.

    max_c_ : str or int
        The identifier of the majority class.

    stats_c_ : dict of str/int : int
        A dictionary in which the number of occurences of each class is
        reported.

    Notes
    -----
    The method is based on [1]_.

    This class supports multi-class.

    References
    ----------
    .. [1] D. Wilson, "Asymptotic Properties of Nearest Neighbor Rules Using
       Edited Data," In IEEE Transactions on Systems, Man, and Cybernetrics,
       vol. 2 (3), pp. 408-421, 1972.

    """

    def __init__(self, estimator, threshold=0.3, mode='maj', cv=5, 
            return_indices=False, random_state=None, verbose=True, n_jobs=-1):

        """Initialisation of Instance Hardness Threshold object.

        Parameters
        ----------
        estimator : sklearn classifier
            Classifier to be used in to estimate instance hardness of the samples.

        threshold : float, optional (default=0.3)
            Threshold to be used when excluding samples (0.01 to 0.99).

        mode: str, optional (default='maj')
            - If 'maj', only samples of the majority class are excluded.
            - If 'all', samples of all classes are excluded.

        cv : int, optional (default=5)
            Number of folds to be used when estimating samples' instance hardness.

        return_indices : bool, optional (default=False)
            Either to return or not the indices which will be selected from
            the majority class.

        random_state : int or None, optional (default=None)
            Seed for random number generation.

        verbose : bool, optional (default=True)
            Boolean to either or not print information about the processing

        n_jobs : int, optional (default=-1)
            The number of thread to open when it is possible.

        Returns
        -------
        None

        """
        super(InstanceHardnessThreshold, self).__init__(
            return_indices=return_indices,
            random_state=random_state,
            verbose=verbose)

        self.estimator = estimator
        self.threshold = threshold

        possible_modes = ('all', 'mode')
        if mode  not in possible_modes:
            raise ValueError('Unknown mode parameter.')
        else:
            self.mode = mode

        self.cv = cv

        self.n_jobs = n_jobs

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
        # Check the consistency of X and y
        X, y = check_X_y(X, y)

        super(InstanceHardnessThreshold, self).fit(X, y)

        return self

    def transform(self, X, y):
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
        # Check the consistency of X and y
	X, y = check_X_y(X, y)

        super(InstanceHardnessThreshold, self).transform(X, y)


        skf = StratifiedKFold(y, n_folds=self.cv, shuffle=False, 
                random_state=self.rs_)


        probabilities = np.zeros(y.shape[0], dtype=float)

        for train_index, test_index in skf:
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            self.estimator.fit(X_train, y_train)

            probs = self.estimator.predict_proba(X_test, y_test)
            probabilities[test_index] = [\
                    probs[l,np.where(self.estimator.classes_ == c)[0][0]] \
                    for l, c in enumerate(yval)]

        if self.mode == 'all':
            mask = probabilities >= self.threshold
        elif self.mode == 'maj':
            mask = np.logical_or(probabilities >= self.threshold, y == self.min_c_)

        X = X[mask].copy()
        y = y[mask].copy()

        # If we need to offer support for the indices
        if self.return_indices:
            idx_under = np.nonzero(mask)[0]
            return X_resampled, y_resampled, idx_under
        else:
            return X_resampled, y_resampled
