"""
Base class for the ensemble method.
"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import numpy as np

from sklearn.preprocessing import label_binarize

from ..base import BaseSampler
from ..utils import check_sampling_strategy


class BaseEnsembleSampler(BaseSampler):
    """Base class for ensemble algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _sampling_type = 'ensemble'

    def fit_resample(self, X, y):
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
        self._deprecate_ratio()
        # Ensemble are a bit specific since they are returning an array of
        # resampled arrays.
        X, y, binarize_y = self._check_X_y(X, y)

        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, self._sampling_type)

        output = self._fit_resample(X, y)

        if binarize_y:
            y_resampled = output[1]
            classes = np.unique(y)
            y_resampled_encoded = np.array(
                [label_binarize(batch_y, classes) for batch_y in y_resampled])
            if len(output) == 2:
                return output[0], y_resampled_encoded
            return output[0], y_resampled_encoded, output[2]
        return output
