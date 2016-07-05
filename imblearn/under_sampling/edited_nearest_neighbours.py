"""Class to perform under-sampling based on the edited nearest neighbour
method."""
from __future__ import print_function
from __future__ import division

import numpy as np

from collections import Counter

from scipy.stats import mode

from sklearn.neighbors import NearestNeighbors

from ..base import SamplerMixin


SEL_KIND = ('all', 'mode')


class EditedNearestNeighbours(SamplerMixin):
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

    size_ngh : int, optional (default=3)
        Size of the neighbourhood to consider to compute the average
        distance to the minority point samples.

    kind_sel : str, optional (default='all')
        Strategy to use in order to exclude samples.

        - If 'all', all neighbours will have to agree with the samples of
        interest to not be excluded.
        - If 'mode', the majority vote of the neighbours will be used in
        order to exclude a sample.

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

    X_shape_ : tuple of int
        Shape of the data `X` during fitting.

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

    def __init__(self, return_indices=False, random_state=None,
                 size_ngh=3, kind_sel='all', n_jobs=-1):
        super(EditedNearestNeighbours, self).__init__()
        self.return_indices = return_indices
        self.random_state = random_state
        self.size_ngh = size_ngh
        self.kind_sel = kind_sel
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
            idx_under = np.nonzero(y == self.min_c_)[0]

        # Create a k-NN to fit the whole data
        nn_obj = NearestNeighbors(n_neighbors=self.size_ngh + 1,
                                  n_jobs=self.n_jobs)
        # Fit the data
        nn_obj.fit(X)

        # Loop over the other classes under picking at random
        for key in self.stats_c_.keys():

            # If the minority class is up, skip it
            if key == self.min_c_:
                continue

            # Get the sample of the current class
            sub_samples_x = X[y == key]
            sub_samples_y = y[y == key]

            # Find the NN for the current class
            nnhood_idx = nn_obj.kneighbors(sub_samples_x,
                                           return_distance=False)[:, 1:]

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
            sel_x = np.squeeze(sub_samples_x[np.nonzero(nnhood_bool), :])
            sel_y = sub_samples_y[np.nonzero(nnhood_bool)]

            # If we need to offer support for the indices selected
            if self.return_indices:
                idx_tmp = np.nonzero(y == key)[0][np.nonzero(nnhood_bool)]
                idx_under = np.concatenate((idx_under, idx_tmp), axis=0)

            X_resampled = np.concatenate((X_resampled, sel_x), axis=0)
            y_resampled = np.concatenate((y_resampled, sel_y), axis=0)

        self.logger.info('Under-sampling performed: %s', Counter(
            y_resampled))

        # Check if the indices of the samples selected should be returned too
        if self.return_indices:
            # Return the indices of interest
            return X_resampled, y_resampled, idx_under
        else:
            return X_resampled, y_resampled


class RepeatedEditedNearestNeighbours(SamplerMixin):
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

    size_ngh : int, optional (default=3)
        Size of the neighbourhood to consider to compute the average
        distance to the minority point samples.

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

    References
    ----------
    .. [1] I. Tomek, "An Experiment with the Edited Nearest-Neighbor
       Rule," IEEE Transactions on Systems, Man, and Cybernetics, vol. 6(6),
       pp. 448-452, June 1976.

    """

    def __init__(self, return_indices=False, random_state=None,
                 size_ngh=3, max_iter=100, kind_sel='all', n_jobs=-1):
        super(RepeatedEditedNearestNeighbours, self).__init__()
        self.return_indices = return_indices
        self.random_state = random_state
        self.size_ngh = size_ngh
        self.kind_sel = kind_sel
        self.n_jobs = n_jobs
        self.max_iter = max_iter
        self.enn_ = EditedNearestNeighbours(
            return_indices=self.return_indices,
            random_state=self.random_state,
            size_ngh=self.size_ngh,
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

        X_, y_ = X.copy(), y.copy()

        if self.return_indices:
            idx_under = np.arange(X.shape[0], dtype=int)

        prev_len = y.shape[0]

        for n_iter in range(self.max_iter):

            self.logger.debug('Apply ENN iteration #%s', n_iter + 1)

            prev_len = y_.shape[0]
            if self.return_indices:
                X_, y_, idx_ = self.enn_.fit_sample(X_, y_)
                idx_under = idx_under[idx_]
            else:
                X_, y_ = self.enn_.fit_sample(X_, y_)

            if prev_len == y_.shape[0]:
                break

        self.logger.info('Under-sampling performed: %s', Counter(y_))

        X_resampled, y_resampled = X_, y_

        # Check if the indices of the samples selected should be returned too
        if self.return_indices:
            # Return the indices of interest
            return X_resampled, y_resampled, idx_under
        else:
            return X_resampled, y_resampled
