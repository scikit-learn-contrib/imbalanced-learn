"""
Base class for the ensemble method.
"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import numpy as np

from sklearn.preprocessing import label_binarize
from sklearn.utils import check_X_y
from sklearn.utils.validation import check_is_fitted

from ..base import BaseSampler
from ..utils import check_target_type


class BaseEnsembleSampler(BaseSampler):
    """Base class for ensemble algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _sampling_type = 'ensemble'

    def sample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like, shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : {ndarray, sparse matrix}, shape \
(n_subset, n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_subset, n_samples_new)
            The corresponding label of `X_resampled`

        """
        # Ensemble are a bit specific since they are returning an array of
        # resampled arrays.
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'])

        check_is_fitted(self, 'ratio_')
        self._check_X_y(X, y)

        output = self._sample(X, y)

        if binarize_y:
            y_resampled = output[1]
            classes = np.unique(y)
            y_resampled_encoded = np.array([label_binarize(batch_y, classes)
                                            for batch_y in y_resampled])
            if len(output) == 2:
                return output[0], y_resampled_encoded
            else:
                return output[0], y_resampled_encoded, output[2]
        else:
            return output
