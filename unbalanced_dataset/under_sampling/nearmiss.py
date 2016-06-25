"""Class to perform under-sampling based on nearmiss methods."""
from __future__ import print_function
from __future__ import division

import numpy as np

from collections import Counter

from sklearn.utils import check_X_y
from sklearn.neighbors import NearestNeighbors

from .under_sampler import UnderSampler


class NearMiss(UnderSampler):
    """Class to perform under-sampling based on NearMiss methods.

    Parameters
    ----------
    ratio : str or float, optional (default='auto')
        If 'auto', the ratio will be defined automatically to balance
        the dataset. Otherwise, the ratio is defined as the number
        of samples in the minority class over the the number of samples
        in the majority class.

    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly
        selected from the majority class.

    random_state : int or None, optional (default=None)
        Seed for random number generation.

    verbose : bool, optional (default=True)
        Whether or not to print information about the processing.

    version : int, optional (default=1)
        Version of the NearMiss to use. Possible values
        are 1, 2 or 3.

    size_ngh : int, optional (default=3)
        Size of the neighbourhood to consider to compute the
        average distance to the minority point samples.

    ver3_samp_ngh : int, optional (default=3)
        NearMiss-3 algorithm start by a phase of re-sampling. This
        parameter correspond to the number of neighbours selected
        create the sub_set in which the selection will be performed.

    n_jobs : int, optional (default=-1)
        The number of threads to open if possible.

    **kwargs : keywords
        Parameter to use for the Nearest Neighbours object.

    Attributes
    ----------
    ratio : str or float
        If 'auto', the ratio will be defined automatically to balance
        the dataset. Otherwise, the ratio is defined as the number
        of samples in the minority class over the the number of samples
        in the majority class.

    random state : int or None
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
    The methods are based on [1]_.

    The class support multi-classes.

    References
    ----------
    .. [1] I. Mani, I. Zhang. "kNN approach to unbalanced data distributions:
       a case study involving information extraction," In Proceedings of
       workshop on learning from imbalanced datasets, 2003.

    """

    def __init__(self, ratio='auto', return_indices=False, random_state=None,
                 verbose=True, version=1, size_ngh=3, ver3_samp_ngh=3,
                 n_jobs=-1, **kwargs):
        """Initialisation of clustering centroids object.

        Parameters
        ----------
        ratio : str or float, optional (default='auto')
            If 'auto', the ratio will be defined automatically to balance
            the dataset. Otherwise, the ratio is defined as the number
            of samples in the minority class over the the number of samples
            in the majority class.

        return_indices : bool, optional (default=False)
            Whether or not to return the indices of the samples randomly
            selected from the majority class.

        random_state : int or None, optional (default=None)
            Seed for random number generation.

        verbose : bool, optional (default=True)
            Whether or not to print information about the processing.

        version : int, optional (default=1)
            Version of the NearMiss to use. Possible values
            are 1, 2 or 3.

        size_ngh : int, optional (default=3)
            Size of the neighbourhood to consider to compute the
            average distance to the minority point samples.

        ver3_samp_ngh : int, optional (default=3)
            NearMiss-3 algorithm start by a phase of re-sampling. This
            parameter correspond to the number of neighbours selected
            create the sub_set in which the selection will be performed.

        n_jobs : int, optional (default=-1)
            The number of threads to open if possible.

        **kwargs : keywords
            Parameter to use for the Nearest Neighbours object.

        Returns
        -------
        None

        """
        super(NearMiss, self).__init__(ratio=ratio,
                                       return_indices=return_indices,
                                       random_state=random_state,
                                       verbose=verbose)

        # Assign the parameter of the element of this class
        # Check that the version asked is implemented
        if not (version == 1 or version == 2 or version == 3):
            raise ValueError('UnbalancedData.NearMiss: there is only 3 '
                             'versions available with parameter version=1/2/3')

        self.version = version
        self.size_ngh = size_ngh
        self.ver3_samp_ngh = ver3_samp_ngh
        self.n_jobs = n_jobs
        self.kwargs = kwargs

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

        super(NearMiss, self).fit(X, y)

        return self

    def _selection_dist_based(self, X, y, dist_vec, num_samples, key,
                              sel_strategy='nearest'):
        """Select the appropriate samples depending of the strategy selected.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Original samples.

        y : ndarray, shape (n_samples, )
            Associated label to X.

        dist_vec : ndarray, shape (n_samples, )
            The distance matrix to the nearest neigbour.

        num_samples: int
            The desired number of samples to select.

        key : str or int,
            The target class.

        sel_strategy : str, optional (default='nearest')
            Strategy to select the samples. Either 'nearest' or 'farthest'

        Returns
        -------
        X_sel : ndarray, shape (num_samples, n_features)
            Selected samples.

        y_sel : ndarray, shape (num_samples, )
            The associated label.

        idx_sel : ndarray, shape (num_samples, )
            The list of the indices of the selected samples.

        """

        # Compute the distance considering the farthest neighbour
        dist_avg_vec = np.sum(dist_vec[:, -self.size_ngh:], axis=1)

        # Sort the list of distance and get the index
        if sel_strategy == 'nearest':
            sort_way = False
        elif sel_strategy == 'farthest':
            sort_way = True
        else:
            raise NotImplementedError

        sorted_idx = sorted(range(len(dist_avg_vec)),
                            key=dist_avg_vec.__getitem__,
                            reverse=sort_way)

        # Select the desired number of samples
        sel_idx = sorted_idx[:num_samples]

        return (X[y == key][sel_idx], y[y == key][sel_idx],
                np.nonzero(y == key)[0][sel_idx])

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

        super(NearMiss, self).sample(X, y)

        # Start with the minority class
        X_min = X[y == self.min_c_]
        y_min = y[y == self.min_c_]

        # All the minority class samples will be preserved
        X_resampled = X_min.copy()
        y_resampled = y_min.copy()

        # Compute the number of cluster needed
        if self.ratio == 'auto':
            num_samples = self.stats_c_[self.min_c_]
        else:
            num_samples = int(self.stats_c_[self.min_c_] / self.ratio)

        # If we need to offer support for the indices
        if self.return_indices:
            idx_under = np.nonzero(y == self.min_c_)[0]

        # For each element of the current class, find the set of NN
        # of the minority class
        # Call the constructor of the NN
        nn_obj = NearestNeighbors(n_neighbors=self.size_ngh,
                                  n_jobs=self.n_jobs,
                                  **self.kwargs)

        # Fit the minority class since that we want to know the distance
        # to these point
        nn_obj.fit(X[y == self.min_c_])

        # Loop over the other classes under picking at random
        for key in self.stats_c_.keys():

            # If the minority class is up, skip it
            if key == self.min_c_:
                continue

            # Get the samples corresponding to the current class
            sub_samples_x = X[y == key]
            sub_samples_y = y[y == key]

            if self.version == 1:
                # Find the NN
                dist_vec, idx_vec = nn_obj.kneighbors(
                    sub_samples_x,
                    n_neighbors=self.size_ngh)

                # Select the right samples
                sel_x, sel_y, idx_tmp = self._selection_dist_based(
                    X,
                    y,
                    dist_vec,
                    num_samples,
                    key,
                    sel_strategy='nearest')

            elif self.version == 2:
                # Find the NN
                dist_vec, idx_vec = nn_obj.kneighbors(
                    sub_samples_x,
                    n_neighbors=self.stats_c_[self.min_c_])

                # Select the right samples
                sel_x, sel_y, idx_tmp = self._selection_dist_based(
                    X,
                    y,
                    dist_vec,
                    num_samples,
                    key,
                    sel_strategy='nearest')

            elif self.version == 3:
                # We need a new NN object to fit the current class
                nn_obj_cc = NearestNeighbors(n_neighbors=self.ver3_samp_ngh,
                                             n_jobs=self.n_jobs,
                                             **self.kwargs)
                nn_obj_cc.fit(sub_samples_x)

                # Find the set of NN to the minority class
                dist_vec, idx_vec = nn_obj_cc.kneighbors(X_min)

                # Create the subset containing the samples found during the NN
                # search. Linearize the indexes and remove the double values
                idx_vec = np.unique(idx_vec.reshape(-1))

                # Create the subset
                sub_samples_x = sub_samples_x[idx_vec, :]
                sub_samples_y = sub_samples_y[idx_vec]

                # Compute the NN considering the current class
                dist_vec, idx_vec = nn_obj.kneighbors(
                    sub_samples_x,
                    n_neighbors=self.size_ngh)

                sel_x, sel_y, idx_tmp = self._selection_dist_based(
                    X,
                    y,
                    dist_vec,
                    num_samples,
                    key,
                    sel_strategy='farthest')
            else:
                raise NotImplementedError

            # If we need to offer support for the indices selected
            if self.return_indices:
                idx_under = np.concatenate((idx_under, idx_tmp), axis=0)

            X_resampled = np.concatenate((X_resampled, sel_x), axis=0)
            y_resampled = np.concatenate((y_resampled, sel_y), axis=0)

        if self.verbose:
            print("Under-sampling performed: {}".format(Counter(y_resampled)))

        # Check if the indices of the samples selected should be returned too
        if self.return_indices:
            # Return the indices of interest
            return X_resampled, y_resampled, idx_under
        else:
            return X_resampled, y_resampled
