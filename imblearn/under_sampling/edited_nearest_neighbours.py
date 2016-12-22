"""Class to perform under-sampling based on the edited nearest neighbour
method."""
from __future__ import division, print_function

from collections import Counter

import numpy as np
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.base import KNeighborsMixin

from ..base import BaseMulticlassSampler

SEL_KIND = ('all', 'mode')


class EditedNearestNeighbours(BaseMulticlassSampler):
    """Class to perform under-sampling based on the edited nearest neighbour
    method.

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
        If object, size of the neighbourhood to consider to compute the average
        distance to the minority point samples.
        If object, an estimator that inherits from
        `sklearn.neighbors.base.KNeighborsMixin` that will be used to find
        the k_neighbors.

    kind_sel : str, optional (default='all')
        Strategy to use in order to exclude samples.

        - If 'all', all neighbours will have to agree with the samples of
        interest to not be excluded.
        - If 'mode', the majority vote of the neighbours will be used in
        order to exclude a sample.

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
    The method is based on [1]_.

    This class supports multi-class.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import \
    EditedNearestNeighbours # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> enn = EditedNearestNeighbours(random_state=42)
    >>> X_res, y_res = enn.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({1: 883, 0: 100})

    References
    ----------
    .. [1] D. Wilson, "Asymptotic Properties of Nearest Neighbor Rules Using
       Edited Data," In IEEE Transactions on Systems, Man, and Cybernetrics,
       vol. 2 (3), pp. 408-421, 1972.

    """

    def __init__(self,
                 return_indices=False,
                 random_state=None,
                 size_ngh=None,
                 n_neighbors=3,
                 kind_sel='all',
                 n_jobs=1):
        super(EditedNearestNeighbours, self).__init__(
            random_state=random_state)
        self.return_indices = return_indices
        self.size_ngh = size_ngh
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Private function to create the NN estimator"""

        if isinstance(self.n_neighbors, int):
            self.nn_ = NearestNeighbors(
                n_neighbors=self.n_neighbors + 1, n_jobs=self.n_jobs)
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

        super(EditedNearestNeighbours, self).fit(X, y)

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

        if self.kind_sel not in SEL_KIND:
            raise NotImplementedError

        # Start with the minority class
        X_min = X[y == self.min_c_]
        y_min = y[y == self.min_c_]

        # All the minority class samples will be preserved
        X_resampled = X_min.copy()
        y_resampled = y_min.copy()

        # If we need to offer support for the indices
        if self.return_indices:
            idx_under = np.flatnonzero(y == self.min_c_)

        # Fit the data
        self.nn_.fit(X)

        # Loop over the other classes under picking at random
        for key in self.stats_c_.keys():

            # If the minority class is up, skip it
            if key == self.min_c_:
                continue

            # Get the sample of the current class
            sub_samples_x = X[y == key]
            sub_samples_y = y[y == key]

            # Find the NN for the current class
            nnhood_idx = self.nn_.kneighbors(
                sub_samples_x, return_distance=False)[:, 1:]

            # Get the label of the corresponding to the index
            nnhood_label = y[nnhood_idx]

            # Check which one are the same label than the current class
            # Make the majority vote
            if self.kind_sel == 'mode':
                nnhood_label, _ = mode(nnhood_label, axis=1)
                nnhood_bool = (np.ravel(nnhood_label) == sub_samples_y)
            elif self.kind_sel == 'all':
                nnhood_label = (nnhood_label == key)
                nnhood_bool = np.all(nnhood_label, axis=1)
            else:
                raise NotImplementedError

            # Get the samples which agree all together
            sel_x = sub_samples_x[np.flatnonzero(nnhood_bool), :]
            sel_y = sub_samples_y[np.flatnonzero(nnhood_bool)]

            # If we need to offer support for the indices selected
            if self.return_indices:
                idx_tmp = np.flatnonzero(y == key)[np.flatnonzero(nnhood_bool)]
                idx_under = np.concatenate((idx_under, idx_tmp), axis=0)

            self.logger.debug('Shape of the selected feature: %s', sel_x.shape)
            self.logger.debug('Shape of current features: %s',
                              X_resampled.shape)

            X_resampled = np.concatenate((X_resampled, sel_x), axis=0)
            y_resampled = np.concatenate((y_resampled, sel_y), axis=0)

        self.logger.info('Under-sampling performed: %s', Counter(y_resampled))

        # Check if the indices of the samples selected should be returned too
        if self.return_indices:
            # Return the indices of interest
            return X_resampled, y_resampled, idx_under
        else:
            return X_resampled, y_resampled


class RepeatedEditedNearestNeighbours(BaseMulticlassSampler):
    """Class to perform under-sampling based on the repeated edited nearest
    neighbour method.

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
        If int, size of the neighbourhood to consider to compute the average
        distance to the minority point samples.
        If object, an estimator that inherits from
        `sklearn.neighbors.base.KNeighborsMixin` that will be used to find
        the k_neighbors.

    kind_sel : str, optional (default='all')
        Strategy to use in order to exclude samples.

        - If 'all', all neighbours will have to agree with the samples of
        interest to not be excluded.
        - If 'mode', the majority vote of the neighbours will be used in
        order to exclude a sample.

    n_jobs : int, optional (default=-1)
        The number of thread to open when it is possible.

    Attributes
    ----------
    min_c_ : str or int
        The identifier of the minority class.

    max_c_ : str or int
        The identifier of the majority class.

    stats_c_ : dict of str/int : int
        A dictionary in which the number of occurences of each class is
        reported.

    max_iter : int, optional (default=100)
        Maximum number of iterations of the edited nearest neighbours
        algorithm for a single run.

    X_shape_ : tuple of int
        Shape of the data `X` during fitting.

    Notes
    -----
    The method is based on [1]_.

    This class supports multi-class.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import \
    RepeatedEditedNearestNeighbours # doctest : +NORMALIZE_WHITESPACE
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
    .. [1] I. Tomek, "An Experiment with the Edited Nearest-Neighbor
       Rule," IEEE Transactions on Systems, Man, and Cybernetics, vol. 6(6),
       pp. 448-452, June 1976.

    """

    def __init__(self,
                 return_indices=False,
                 random_state=None,
                 size_ngh=None,
                 n_neighbors=3,
                 max_iter=100,
                 kind_sel='all',
                 n_jobs=-1):
        super(RepeatedEditedNearestNeighbours, self).__init__(
            random_state=random_state)
        self.return_indices = return_indices
        self.size_ngh = size_ngh
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.n_jobs = n_jobs
        self.max_iter = max_iter

    def _validate_estimator(self):
        """Private function to create the NN estimator"""

        self.enn_ = EditedNearestNeighbours(
            return_indices=self.return_indices,
            random_state=self.random_state,
            n_neighbors=self.n_neighbors,
            kind_sel=self.kind_sel,
            n_jobs=self.n_jobs)

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

        super(RepeatedEditedNearestNeighbours, self).fit(X, y)

        self._validate_estimator()

        self.enn_.fit(X, y)

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

        if self.kind_sel not in SEL_KIND:
            raise NotImplementedError

        if self.max_iter < 2:
            raise ValueError('max_iter must be greater than 1.')

        X_, y_ = X, y

        if self.return_indices:
            idx_under = np.arange(X.shape[0], dtype=int)

        prev_len = y.shape[0]

        for n_iter in range(self.max_iter):

            self.logger.debug('Apply ENN iteration #%s', n_iter + 1)

            prev_len = y_.shape[0]
            if self.return_indices:
                X_enn, y_enn, idx_enn = self.enn_.fit_sample(X_, y_)
            else:
                X_enn, y_enn = self.enn_.fit_sample(X_, y_)

            # Check the stopping criterion
            # 1. If there is no changes for the vector y
            # 2. If the number of samples in the other class become inferior to
            # the number of samples in the majority class
            # 3. If one of the class is disappearing

            # Case 1
            b_conv = (prev_len == y_enn.shape[0])

            # Case 2
            stats_enn = Counter(y_enn)
            self.logger.debug('Current ENN stats: %s', stats_enn)
            # Get the number of samples in the non-minority classes
            count_non_min = np.array([
                val for val, key in zip(stats_enn.values(), stats_enn.keys())
                if key != self.min_c_
            ])
            self.logger.debug('Number of samples in the non-majority'
                              ' classes: %s', count_non_min)
            # Check the minority stop to be the minority
            b_min_bec_maj = np.any(count_non_min < self.stats_c_[self.min_c_])

            # Case 3
            b_remove_maj_class = (len(stats_enn) < len(self.stats_c_))

            if b_conv or b_min_bec_maj or b_remove_maj_class:
                # If this is a normal convergence, get the last data
                if b_conv:
                    if self.return_indices:
                        X_, y_, = X_enn, y_enn
                        idx_under = idx_under[idx_enn]
                    else:
                        X_, y_, = X_enn, y_enn
                # Log the variables to explain the stop of the algorithm
                self.logger.debug('RENN converged: %s', b_conv)
                self.logger.debug('RENN minority become majority: %s',
                                  b_min_bec_maj)
                self.logger.debug('RENN remove one class: %s',
                                  b_remove_maj_class)
                break

            # Update the data for the next iteration
            X_, y_, = X_enn, y_enn
            if self.return_indices:
                idx_under = idx_under[idx_enn]

        self.logger.info('Under-sampling performed: %s', Counter(y_))

        X_resampled, y_resampled = X_, y_

        # Check if the indices of the samples selected should be returned too
        if self.return_indices:
            # Return the indices of interest
            return X_resampled, y_resampled, idx_under
        else:
            return X_resampled, y_resampled


class AllKNN(BaseMulticlassSampler):
    """Class to perform under-sampling based on the AllKNN method.

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
        If int, size of the neighbourhood to consider to compute the average
        distance to the minority point samples.
        If object, an estimator that inherits from
        `sklearn.neighbors.base.KNeighborsMixin` that will be used to find
        the k_neighbors.

    kind_sel : str, optional (default='all')
        Strategy to use in order to exclude samples.

        - If 'all', all neighbours will have to agree with the samples of
        interest to not be excluded.
        - If 'mode', the majority vote of the neighbours will be used in
        order to exclude a sample.

    n_jobs : int, optional (default=-1)
        The number of thread to open when it is possible.

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
    The method is based on [1]_.

    This class supports multi-class.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import \
    AllKNN # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> allknn = AllKNN(random_state=42)
    >>> X_res, y_res = allknn.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({1: 883, 0: 100})

    References
    ----------
    .. [1] I. Tomek, "An Experiment with the Edited Nearest-Neighbor
       Rule," IEEE Transactions on Systems, Man, and Cybernetics, vol. 6(6),
       pp. 448-452, June 1976.

    """

    def __init__(self,
                 return_indices=False,
                 random_state=None,
                 size_ngh=None,
                 n_neighbors=3,
                 kind_sel='all',
                 n_jobs=-1):
        super(AllKNN, self).__init__(random_state=random_state)
        self.return_indices = return_indices
        self.size_ngh = size_ngh
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Private function to create the NN estimator"""

        self.enn_ = EditedNearestNeighbours(
            return_indices=self.return_indices,
            random_state=self.random_state,
            n_neighbors=self.n_neighbors,
            kind_sel=self.kind_sel,
            n_jobs=self.n_jobs)

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
        super(AllKNN, self).fit(X, y)

        self._validate_estimator()

        self.enn_.fit(X, y)

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

        if self.kind_sel not in SEL_KIND:
            raise NotImplementedError

        X_, y_ = X, y

        if self.return_indices:
            idx_under = np.arange(X.shape[0], dtype=int)

        for curr_size_ngh in range(1, self.enn_.nn_.n_neighbors):
            self.logger.debug('Apply ENN n_neighbors #%s', curr_size_ngh)
            # updating ENN size_ngh
            self.enn_.n_neighbors = curr_size_ngh

            if self.return_indices:
                X_enn, y_enn, idx_enn = self.enn_.fit_sample(X_, y_)
            else:
                X_enn, y_enn = self.enn_.fit_sample(X_, y_)

            # Check the stopping criterion
            # 1. If the number of samples in the other class become inferior to
            # the number of samples in the majority class
            # 2. If one of the class is disappearing
            # Case 1
            stats_enn = Counter(y_enn)
            self.logger.debug('Current ENN stats: %s', stats_enn)
            # Get the number of samples in the non-minority classes
            count_non_min = np.array([
                val for val, key in zip(stats_enn.values(), stats_enn.keys())
                if key != self.min_c_
            ])
            self.logger.debug('Number of samples in the non-majority'
                              ' classes: %s', count_non_min)
            # Check the minority stop to be the minority
            b_min_bec_maj = np.any(count_non_min < self.stats_c_[self.min_c_])

            # Case 2
            b_remove_maj_class = (len(stats_enn) < len(self.stats_c_))

            if b_min_bec_maj or b_remove_maj_class:
                # Log the variables to explain the stop of the algorithm
                self.logger.debug('AllKNN minority become majority: %s',
                                  b_min_bec_maj)
                self.logger.debug('AllKNN remove one class: %s',
                                  b_remove_maj_class)
                break

            # Update the data for the next iteration
            X_, y_, = X_enn, y_enn
            if self.return_indices:
                idx_under = idx_under[idx_enn]

        self.logger.info('Under-sampling performed: %s', Counter(y_))

        X_resampled, y_resampled = X_, y_

        # Check if the indices of the samples selected should be returned too
        if self.return_indices:
            # Return the indices of interest
            return X_resampled, y_resampled, idx_under
        else:
            return X_resampled, y_resampled
