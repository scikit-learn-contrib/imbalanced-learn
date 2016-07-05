"""Class to perform under-sampling using easy ensemble."""
from __future__ import print_function

import numpy as np

from ..base import SamplerMixin
from ..under_sampling import RandomUnderSampler


class EasyEnsemble(SamplerMixin):
    """Create an ensemble sets by iteratively applying random under-sampling.

    This method iteratively select a random subset and make an ensemble of the
    different sets.

    Parameters
    ----------
    ratio : str or float, optional (default='auto')
        If 'auto', the ratio will be defined automatically to balance
        the dataset. Otherwise, the ratio is defined as the number
        of samples in the minority class over the the number of samples
        in the majority class.

    return_indices : bool, optional (default=True)
        Whether or not to return the indices of the samples randomly
        selected from the majority class.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    replacement : bool, optional (default=False)
        Whether or not to sample randomly with replacement or not.

    n_subsets : int, optional (default=10)
        Number of subsets to generate.

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
    The method is described in [1]_.

    References
    ----------
    .. [1] X. Y. Liu, J. Wu and Z. H. Zhou, "Exploratory Undersampling for
       Class-Imbalance Learning," in IEEE Transactions on Systems, Man, and
       Cybernetics, Part B (Cybernetics), vol. 39, no. 2, pp. 539-550,
       April 2009.

    """

    def __init__(self, ratio='auto', return_indices=False,
                 random_state=None, replacement=False, n_subsets=10):
        super(EasyEnsemble, self).__init__(ratio=ratio)
        self.return_indices = return_indices
        self.random_state = random_state
        self.replacement = replacement
        self.n_subsets = n_subsets

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
        X_resampled : ndarray, shape (n_subset, n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_subset, n_samples_new)
            The corresponding label of `X_resampled`

        idx_under : ndarray, shape (n_subset, n_samples, )
            If `return_indices` is `True`, a boolean array will be returned
            containing the which samples have been selected.

        """

        X_resampled = []
        y_resampled = []
        if self.return_indices:
            idx_under = []

        for s in range(self.n_subsets):
            self.logger.debug('Creation of the set #%s', s)

            # Create the object for random under-sampling
            rus = RandomUnderSampler(ratio=self.ratio,
                                     return_indices=self.return_indices,
                                     random_state=self.random_state,
                                     replacement=self.replacement)
            if self.return_indices:
                sel_x, sel_y, sel_idx = rus.fit_sample(X, y)
            else:
                sel_x, sel_y = rus.fit_sample(X, y)

            X_resampled.append(sel_x)
            y_resampled.append(sel_y)
            if self.return_indices:
                idx_under.append(sel_idx)

        if self.return_indices:
            return (np.array(X_resampled), np.array(y_resampled),
                    np.array(idx_under))
        else:
            return np.array(X_resampled), np.array(y_resampled)
