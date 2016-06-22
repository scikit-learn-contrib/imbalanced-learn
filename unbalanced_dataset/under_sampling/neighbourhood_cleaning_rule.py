"""Class performing under-sampling based on the neighbourhood cleaning rule."""
from __future__ import print_function
from __future__ import division

import numpy as np

from collections import Counter

from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_X_y

from .under_sampler import UnderSampler


class NeighbourhoodCleaningRule(UnderSampler):
    """Class performing under-sampling based on the neighbourhood cleaning
    rule.

    Parameters
    ----------
    return_indices : bool, optional (default=False)
        Either to return or not the indices which will be selected from
        the majority class.

    random_state : int or None, optional (default=None)
        Seed for random number generation.

    verbose : bool, optional (default=True)
        Boolean to either or not print information about the processing

    size_ngh : int, optional (default=3)
        Size of the neighbourhood to consider in order to make
        the comparison between each samples and their NN.

    n_jobs : int, optional (default=-1)
        The number of thread to open when it is possible.

    **kwargs : keywords
        Parameter to use for the Neareast Neighbours object.

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
    This class support multi-class.

    References
    ----------
    .. [1] J. Laurikkala, "Improving identification of difficult small classes
       by balancing class distribution," Springer Berlin Heidelberg, 2001.

    """

    def __init__(self, return_indices=False, random_state=None, verbose=True,
                 size_ngh=3, n_jobs=-1):
        """Initialisation of NCL object.

        Parameters
        ----------
        return_indices : bool, optional (default=False)
            Either to return or not the indices which will be selected from
            the majority class.

        random_state : int or None, optional (default=None)
            Seed for random number generation.

        verbose : bool, optional (default=True)
            Boolean to either or not print information about the processing

        size_ngh : int, optional (default=3)
            Size of the neighbourhood to consider in order to make
            the comparison between each samples and their NN.

        n_jobs : int, optional (default=-1)
            The number of thread to open when it is possible.

        **kwargs : keywords
            Parameter to use for the Neareast Neighbours object.

        Returns
        -------
        None

        """
        super(NeighbourhoodCleaningRule, self).__init__(
            return_indices=return_indices,
            random_state=random_state,
            verbose=verbose)

        self.size_ngh = size_ngh
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

        super(NeighbourhoodCleaningRule, self).fit(X, y)

        return self

    def sample(self, X, y):
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

        super(NeighbourhoodCleaningRule, self).sample(X, y)

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
        nn_obj = NearestNeighbors(n_neighbors=self.size_ngh,
                                  n_jobs=self.n_jobs)

        # Fit the whole dataset
        nn_obj.fit(X)

        idx_to_exclude = []
        # Loop over the other classes under picking at random
        for key in self.stats_c_.keys():

            # Get the sample of the current class
            sub_samples_x = X[y == key]

            # Get the samples associated
            idx_sub_sample = np.nonzero(y == key)[0]

            # Find the NN for the current class
            nnhood_idx = nn_obj.kneighbors(sub_samples_x,
                                           return_distance=False)

            # Get the label of the corresponding to the index
            nnhood_label = (y[nnhood_idx] == key)

            # Check which one are the same label than the current class
            # Make an AND operation through the three neighbours
            nnhood_bool = np.logical_not(np.all(nnhood_label, axis=1))

            # If the minority class remove the majority samples
            if key == self.min_c_:
                # Get the index to exclude
                idx_to_exclude += nnhood_idx[np.nonzero(
                    nnhood_label[np.nonzero(nnhood_bool)])].tolist()
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
        sel_x = np.squeeze(X[np.nonzero(sel_idx), :])
        sel_y = y[np.nonzero(sel_idx)]

        # If we need to offer support for the indices selected
        if self.return_indices:
            idx_tmp = np.nonzero(sel_idx)[0]
            idx_under = np.concatenate((idx_under, idx_tmp), axis=0)

        X_resampled = np.concatenate((X_resampled, sel_x), axis=0)
        y_resampled = np.concatenate((y_resampled, sel_y), axis=0)

        if self.verbose:
            print("Under-sampling performed: {}" + str(Counter(y_resampled)))

        # Check if the indices of the samples selected should be returned too
        if self.return_indices:
            # Return the indices of interest
            return X_resampled, y_resampled, idx_under
        else:
            return X_resampled, y_resampled
