"""Class to perform under-sampling based on the condensed nearest neighbour
method."""
from __future__ import print_function
from __future__ import division

import numpy as np

from collections import Counter

from sklearn.utils import check_X_y
from sklearn.neighbors import KNeighborsClassifier

from .under_sampler import UnderSampler


class CondensedNearestNeighbour(UnderSampler):
    """Class to perform under-sampling based on the condensed nearest neighbour
    method.

    Parameters
    ----------
    return_indices : bool, optional (default=False)
        Either to return or not the indices which will be selected from
        the majority class.

    random_state : int or None, optional (default=None)
        Seed for random number generation.

    verbose : bool, optional (default=True)
        Boolean to either or not print information about the processing

    size_ngh : int, optional (default=1)
        Size of the neighbourhood to consider to compute the average
        distance to the minority point samples.

    n_seeds_S : int, optional (default=1)
        Number of samples to extract in order to build the set S.

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
    The method is based on [1]_.

    This class supports multi-class.

    References
    ----------
    .. [1] M. Kubat, S. Matwin, "Addressing the curse of imbalanced training
       sets: one-sided selection," In ICML, vol. 97, pp. 179-186, 1997.

    """

    def __init__(self, return_indices=False, random_state=None, verbose=True,
                 size_ngh=1, n_seeds_S=1, n_jobs=-1, **kwargs):
        """Initialisation of CNN object.

        Parameters
        ----------
        return_indices : bool, optional (default=False)
            Either to return or not the indices which will be selected from
            the majority class.

        random_state : int or None, optional (default=None)
            Seed for random number generation.

        verbose : bool, optional (default=True)
            Boolean to either or not print information about the processing

        size_ngh : int, optional (default=1)
            Size of the neighbourhood to consider to compute the average
            distance to the minority point samples.

        n_seeds_S : int, optional (default=1)
            Number of samples to extract in order to build the set S.

        n_jobs : int, optional (default=-1)
            The number of thread to open when it is possible.

        **kwargs : keywords
            Parameter to use for the Neareast Neighbours object.

        Returns
        -------
        None

        """
        super(CondensedNearestNeighbour, self).__init__(
            return_indices=return_indices,
            random_state=random_state,
            verbose=verbose)

        self.size_ngh = size_ngh
        self.n_seeds_S = n_seeds_S
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

        super(CondensedNearestNeighbour, self).fit(X, y)

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

        super(CondensedNearestNeighbour, self).sample(X, y)

        # Start with the minority class
        X_min = X[y == self.min_c_]
        y_min = y[y == self.min_c_]

        # All the minority class samples will be preserved
        X_resampled = X_min.copy()
        y_resampled = y_min.copy()

        # If we need to offer support for the indices
        if self.return_indices:
            idx_under = np.nonzero(y == self.min_c_)[0]

        # Loop over the other classes under picking at random
        for key in self.stats_c_.keys():

            # If the minority class is up, skip it
            if key == self.min_c_:
                continue

            # Randomly get one sample from the majority class
            np.random.seed(self.random_state)
            # Generate the index to select
            idx_maj_sample = np.random.randint(low=0, high=self.stats_c_[key],
                                               size=self.n_seeds_S)
            maj_sample = X[y == key][idx_maj_sample]

            # Create the set C - One majority samples and all minority
            C_x = np.append(X_min, maj_sample, axis=0)
            C_y = np.append(y_min, np.array([key] * self.n_seeds_S))

            # Create the set S - all majority samples
            S_x = X[y == key]
            S_y = y[y == key]

            # Create a k-NN classifier
            knn = KNeighborsClassifier(n_neighbors=self.size_ngh,
                                       n_jobs=self.n_jobs,
                                       **self.kwargs)

            # Fit C into the knn
            knn.fit(C_x, C_y)

            good_classif_label = idx_maj_sample.copy()
            # Check each sample in S if we keep it or drop it
            for idx_sam, (x_sam, y_sam) in enumerate(zip(S_x, S_y)):

                # Do not select sample which are already well classified
                if idx_sam in good_classif_label:
                    continue

                # Classify on S
                pred_y = knn.predict(x_sam.reshape(1, -1))

                # If the prediction do not agree with the true label
                # append it in C_x
                if y_sam != pred_y:
                    # Keep the index for later
                    idx_maj_sample = np.append(idx_maj_sample, idx_sam)

                    # Update C
                    C_x = np.append(X_min, X[y == key][idx_maj_sample], axis=0)
                    C_y = np.append(y_min, np.array([key] *
                                                    idx_maj_sample.size))

                    # Fit C into the knn
                    knn.fit(C_x, C_y)

                    # This experimental to speed up the search
                    # Classify all the element in S and avoid to test the
                    # well classified elements
                    pred_S_y = knn.predict(S_x)
                    good_classif_label = np.unique(
                        np.append(idx_maj_sample,
                                  np.nonzero(pred_S_y == S_y)[0]))

            # Find the misclassified S_y
            sel_x = np.squeeze(S_x[idx_maj_sample, :])
            sel_y = S_y[idx_maj_sample]

            # If we need to offer support for the indices selected
            if self.return_indices:
                idx_under = np.concatenate((idx_under, idx_maj_sample), axis=0)

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
