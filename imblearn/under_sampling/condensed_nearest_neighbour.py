"""Class to perform under-sampling based on the condensed nearest neighbour
method."""
from __future__ import division, print_function

from collections import Counter

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_random_state

from ..base import BaseMulticlassSampler


class CondensedNearestNeighbour(BaseMulticlassSampler):
    """Class to perform under-sampling based on the condensed nearest neighbour
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

    n_neighbors : int or object, optional (default=
        KNeighborsClassifier(n_neighbors=1))
        If int, size of the neighbourhood to consider to compute the average
        distance to the minority point samples.
        If object, an object inherited from
        `sklearn.neigbors.KNeighborsClassifier` should be passed.

    n_seeds_S : int, optional (default=1)
        Number of samples to extract in order to build the set S.

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

    >>> from collections import Counter #doctest: +SKIP
    >>> from sklearn.datasets import fetch_mldata #doctest: +SKIP
    >>> from imblearn.under_sampling import \
    CondensedNearestNeighbour #doctest: +SKIP
    >>> pima = fetch_mldata('diabetes_scale') #doctest: +SKIP
    >>> X, y = pima['data'], pima['target'] #doctest: +SKIP
    >>> print('Original dataset shape {}'.format(Counter(y))) #doctest: +SKIP
    Original dataset shape Counter({1: 500, -1: 268}) #doctest: +SKIP
    >>> cnn = CondensedNearestNeighbour(random_state=42) #doctest: +SKIP
    >>> X_res, y_res = cnn.fit_sample(X, y) #doctest: +SKIP
    >>> print('Resampled dataset shape {}'.format(
    ... Counter(y_res))) #doctest: +SKIP
    Resampled dataset shape Counter({-1: 268, 1: 227}) #doctest: +SKIP

    References
    ----------
    .. [1] P. Hart, "The condensed nearest neighbor rule,"
       In Information Theory, IEEE Transactions on, vol. 14(3),
       pp. 515-516, 1968.

    """

    def __init__(self,
                 return_indices=False,
                 random_state=None,
                 size_ngh=None,
                 n_neighbors=None,
                 n_seeds_S=1,
                 n_jobs=1):
        super(CondensedNearestNeighbour, self).__init__(
            random_state=random_state)
        self.return_indices = return_indices
        self.size_ngh = size_ngh
        self.n_neighbors = n_neighbors
        self.n_seeds_S = n_seeds_S
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Private function to create the NN estimator"""

        if self.n_neighbors is None:
            self.estimator_ = KNeighborsClassifier(
                n_neighbors=1, n_jobs=self.n_jobs)
        elif isinstance(self.n_neighbors, int):
            self.estimator_ = KNeighborsClassifier(
                n_neighbors=self.n_neighbors, n_jobs=self.n_jobs)
        elif isinstance(self.n_neighbors, KNeighborsClassifier):
            self.estimator_ = self.n_neighbors
        else:
            raise ValueError('`n_neighbors` has to be a in or an object'
                             ' inhereited from KNeighborsClassifier.')

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

        super(CondensedNearestNeighbour, self).fit(X, y)

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

        random_state = check_random_state(self.random_state)

        # Start with the minority class
        X_min = X[y == self.min_c_]
        y_min = y[y == self.min_c_]

        # All the minority class samples will be preserved
        X_resampled = X_min.copy()
        y_resampled = y_min.copy()

        # If we need to offer support for the indices
        if self.return_indices:
            idx_under = np.flatnonzero(y == self.min_c_)

        # Loop over the other classes under picking at random
        for key in self.stats_c_.keys():

            # If the minority class is up, skip it
            if key == self.min_c_:
                continue

            # Randomly get one sample from the majority class
            # Generate the index to select
            idx_maj_sample = random_state.randint(
                low=0, high=self.stats_c_[key], size=self.n_seeds_S)
            maj_sample = X[y == key][idx_maj_sample]

            # Create the set C - One majority samples and all minority
            C_x = np.append(X_min, maj_sample, axis=0)
            C_y = np.append(y_min, np.array([key] * self.n_seeds_S))

            # Create the set S - all majority samples
            S_x = X[y == key]
            S_y = y[y == key]

            # Fit C into the knn
            self.estimator_.fit(C_x, C_y)

            good_classif_label = idx_maj_sample.copy()
            # Check each sample in S if we keep it or drop it
            for idx_sam, (x_sam, y_sam) in enumerate(zip(S_x, S_y)):

                # Do not select sample which are already well classified
                if idx_sam in good_classif_label:
                    continue

                # Classify on S
                pred_y = self.estimator_.predict(x_sam.reshape(1, -1))

                # If the prediction do not agree with the true label
                # append it in C_x
                if y_sam != pred_y:
                    # Keep the index for later
                    idx_maj_sample = np.append(idx_maj_sample, idx_sam)

                    # Update C
                    C_x = np.append(X_min, X[y == key][idx_maj_sample], axis=0)
                    C_y = np.append(y_min,
                                    np.array([key] * idx_maj_sample.size))

                    # Fit C into the knn
                    self.estimator_.fit(C_x, C_y)

                    # This experimental to speed up the search
                    # Classify all the element in S and avoid to test the
                    # well classified elements
                    pred_S_y = self.estimator_.predict(S_x)
                    good_classif_label = np.unique(
                        np.append(idx_maj_sample,
                                  np.flatnonzero(pred_S_y == S_y)))

            # Find the misclassified S_y
            sel_x = S_x[idx_maj_sample, :]
            sel_y = S_y[idx_maj_sample]

            # The indexes found are relative to the current class, we need to
            # find the absolute value
            # Build the array with the absolute position
            abs_pos = np.flatnonzero(y == key)
            idx_maj_sample = abs_pos[idx_maj_sample]

            # If we need to offer support for the indices selected
            if self.return_indices:
                idx_under = np.concatenate((idx_under, idx_maj_sample), axis=0)

            X_resampled = np.concatenate((X_resampled, sel_x), axis=0)
            y_resampled = np.concatenate((y_resampled, sel_y), axis=0)

        self.logger.info('Under-sampling performed: %s', Counter(y_resampled))

        # Check if the indices of the samples selected should be returned too
        if self.return_indices:
            # Return the indices of interest
            return X_resampled, y_resampled, idx_under
        else:
            return X_resampled, y_resampled
