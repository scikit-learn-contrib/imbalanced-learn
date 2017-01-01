"""Class performing under-sampling based on the neighbourhood cleaning rule."""
from __future__ import division, print_function

from collections import Counter

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.base import KNeighborsMixin

from ..base import BaseMulticlassSampler


class NeighbourhoodCleaningRule(BaseMulticlassSampler):
    """Class performing under-sampling based on the neighbourhood cleaning
    rule.

    Parameters
    ----------
    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly
        selected from the majority class.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    size_ngh : int, optional (default=None)
        Size of the neighbourhood to consider to compute the average
        distance to the minority point samples.

        NOTE: size_ngh is deprecated from 0.2 and will be replaced in 0.4
        Use ``n_neighbors`` instead.

    n_neighbors : int or object, optional (default=3)
        If int, size of the neighbourhood to consider in order to make
        the comparison between each samples and their NN.
        If object, an estimator that inherits from
        `sklearn.neighbors.base.KNeighborsMixin` that will be used to find
        the k_neighbors.

    n_jobs : int, optional (default=1)
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

    X_shape_ : tuple of int
        Shape of the data `X` during fitting.

    Notes
    -----
    This class support multi-class.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import \
    NeighbourhoodCleaningRule # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> ncr = NeighbourhoodCleaningRule(random_state=42)
    >>> X_res, y_res = ncr.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({1: 891, 0: 100})

    References
    ----------
    .. [1] J. Laurikkala, "Improving identification of difficult small classes
       by balancing class distribution," Springer Berlin Heidelberg, 2001.

    """

    def __init__(self,
                 return_indices=False,
                 random_state=None,
                 size_ngh=None,
                 n_neighbors=3,
                 n_jobs=1):
        super(NeighbourhoodCleaningRule, self).__init__(
            random_state=random_state)
        self.return_indices = return_indices
        self.size_ngh = size_ngh
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Private function to create the NN estimator"""

        if isinstance(self.n_neighbors, int):
            self.nn_ = NearestNeighbors(
                n_neighbors=self.n_neighbors, n_jobs=self.n_jobs)
        elif isinstance(self.n_neighbors, KNeighborsMixin):
            self.nn_ = self.n_neighbors
        else:
            raise ValueError('`n_neighbors` has to be be either int or a'
                             ' subclass of KNeighborsMixin.')

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

        super(NeighbourhoodCleaningRule, self).fit(X, y)

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

        # Start with the minority class
        X_min = X[y == self.min_c_]
        y_min = y[y == self.min_c_]

        # All the minority class samples will be preserved
        X_resampled = X_min.copy()
        y_resampled = y_min.copy()

        # If we need to offer support for the indices
        if self.return_indices:
            idx_under = np.flatnonzero(y == self.min_c_)

        # Fit the whole dataset
        self.nn_.fit(X)

        idx_to_exclude = []
        # Loop over the other classes under picking at random
        for key in self.stats_c_.keys():

            # Get the sample of the current class
            sub_samples_x = X[y == key]

            # Get the samples associated
            idx_sub_sample = np.flatnonzero(y == key)

            # Find the NN for the current class
            nnhood_idx = self.nn_.kneighbors(
                sub_samples_x, return_distance=False)

            # Get the label of the corresponding to the index
            nnhood_label = (y[nnhood_idx] == key)

            # Check which one are the same label than the current class
            # Make an AND operation through the three neighbours
            nnhood_bool = np.logical_not(np.all(nnhood_label, axis=1))

            # If the minority class remove the majority samples
            if key == self.min_c_:
                # Get the index to exclude
                idx_to_exclude += nnhood_idx[np.nonzero(nnhood_label[
                    np.flatnonzero(nnhood_bool)])].tolist()
            else:
                # Get the index to exclude
                idx_to_exclude += idx_sub_sample[np.nonzero(
                    nnhood_bool)].tolist()

        idx_to_exclude = np.unique(idx_to_exclude)

        # Create a vector with the sample to select
        sel_idx = np.ones(y.shape)
        sel_idx[idx_to_exclude] = 0
        # Exclude as well the minority sample since that they will be
        # concatenated later
        sel_idx[y == self.min_c_] = 0

        # Get the samples from the majority classes
        sel_x = X[np.flatnonzero(sel_idx), :]
        sel_y = y[np.flatnonzero(sel_idx)]

        # If we need to offer support for the indices selected
        if self.return_indices:
            idx_tmp = np.flatnonzero(sel_idx)
            idx_under = np.concatenate((idx_under, idx_tmp), axis=0)

        X_resampled = np.concatenate((X_resampled, sel_x), axis=0)
        y_resampled = np.concatenate((y_resampled, sel_y), axis=0)

        self.logger.info('Under-sampling performed: %s', Counter(y_resampled))

        # Check if the indices of the samples selected should be returned too
        if self.return_indices:
            # Return the indices of interest
            return X_resampled, y_resampled, idx_under
        else:
            return X_resampled, y_resampled
