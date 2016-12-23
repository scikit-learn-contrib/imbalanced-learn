"""Class to perform random under-sampling."""
from __future__ import division, print_function

from collections import Counter

import numpy as np
from sklearn.utils import check_random_state

from ..base import BaseMulticlassSampler


class RandomUnderSampler(BaseMulticlassSampler):
    """Class to perform random under-sampling.

    Under-sample the majority class(es) by randomly picking samples
    with or without replacement.

    Parameters
    ----------
    ratio : str or float, optional (default='auto')
        If 'auto', the ratio will be defined automatically to balance
        the dataset. Otherwise, the ratio is defined as the number
        of samples in the minority class over the the number of samples
        in the majority class.

    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly selected
        from the majority class.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    Attributes
    ----------
    min_c_ : str or int
        The identifier of the minority class.

    max_c_ : str or int
        The identifier of the majority class.

    stats_c_ : dict of str/int : int
        A dictionary containing the number of occurences of each class.

    X_shape_ : tuple of int
        Shape of the data `X` during fitting.

    Notes
    -----
    This class supports multi-class.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import \
    RandomUnderSampler # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ...  weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> rus = RandomUnderSampler(random_state=42)
    >>> X_res, y_res = rus.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({0: 100, 1: 100})

    """

    def __init__(self,
                 ratio='auto',
                 return_indices=False,
                 random_state=None,
                 replacement=True):
        super(RandomUnderSampler, self).__init__(
            ratio=ratio, random_state=random_state)
        self.return_indices = return_indices
        self.replacement = replacement

    def _sample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data to be sampled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : ndarray, shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new)
            The corresponding label of `X_resampled`

        idx_under : ndarray, shape (n_samples, )
            If `return_indices` is `True`, an array will be returned
            containing a boolean for each sample to represent whether
            that sample was selected or not.

        """

        random_state = check_random_state(self.random_state)

        # Compute the number of clusters needed
        if self.ratio == 'auto':
            num_samples = self.stats_c_[self.min_c_]
        else:
            num_samples = int(self.stats_c_[self.min_c_] / self.ratio)

        # All the minority class samples will be preserved
        X_resampled = X[y == self.min_c_]
        y_resampled = y[y == self.min_c_]

        # If we need to offer support for the indices
        if self.return_indices:
            idx_under = np.nonzero(y == self.min_c_)[0]

        # Loop over the other classes under-picking at random
        for key in self.stats_c_.keys():

            # If the minority class is up, skip it
            if key == self.min_c_:
                continue

            # Pick some elements at random
            indx = range(np.count_nonzero(y == key))
            indx = random_state.choice(
                indx, size=num_samples, replace=self.replacement)

            # If we need to offer support for the indices selected
            if self.return_indices:
                idx_tmp = np.nonzero(y == key)[0][indx]
                idx_under = np.concatenate((idx_under, idx_tmp), axis=0)

            # Concatenate to the minority class
            X_resampled = np.concatenate(
                (X_resampled, X[y == key][indx]), axis=0)
            y_resampled = np.concatenate(
                (y_resampled, y[y == key][indx]), axis=0)

        self.logger.info('Under-sampling performed: %s', Counter(y_resampled))

        # Check if the indices of the samples selected should be returned as
        # well
        if self.return_indices:
            # Return the indices of interest
            return X_resampled, y_resampled, idx_under
        else:
            return X_resampled, y_resampled
