"""Class to perform random under-sampling."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import division, print_function

from collections import Counter

import numpy as np
from sklearn.utils import check_random_state

from ...base import MultiClassSamplerMixin
from ..base import BaseUnderSampler


class RandomUnderSampler(BaseUnderSampler, MultiClassSamplerMixin):
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

    replacement : boolean, optional (default=False)
        Whether the sample is with (default) or without replacement.

    Attributes
    ----------
    X_shape_ : tuple of int
        Shape of the data `X` during fitting.

    ratio_ : dict
        Dictionary in which the keys are the classes and the values are the
        number of samples to be kept.

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
                 replacement=False):
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

        X_resampled = np.empty((0, X.shape[1]), dtype=X.dtype)
        y_resampled = np.empty((0, ), dtype=y.dtype)
        if self.return_indices:
            idx_under = np.empty((0, ), dtype=int)

        for target_class in np.unique(y):
            if target_class in self.ratio_.keys():
                n_samples = self.ratio_[target_class]
                index_target_class = random_state.choice(
                    range(np.count_nonzero(y == target_class)),
                    size=n_samples,
                    replace=self.replacement)
            else:
                index_target_class = slice(None)

            X_resampled = np.concatenate(
                (X_resampled, X[y == target_class][index_target_class]),
                axis=0)
            y_resampled = np.concatenate(
                (y_resampled, y[y == target_class][index_target_class]),
                axis=0)
            if self.return_indices:
                idx_under = np.concatenate(
                    (idx_under, np.flatnonzero(y == target_class)[
                        index_target_class]), axis=0)

        self.logger.info('Under-sampling performed: %s', Counter(y_resampled))

        if self.return_indices:
            return X_resampled, y_resampled, idx_under
        else:
            return X_resampled, y_resampled
