"""
Base class for the over-sampling method.
"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from sklearn.utils import check_X_y

from ..base import BaseSampler


class BaseOverSampler(BaseSampler):
    """Base class for over-sampling algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    _sampling_type = 'over-sampling'

    def fit(self, X, y):
        """Find the classes statistics before to perform sampling.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like, shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        self : object,
            Return self.

        Notes
        -----
        Over-samplers do not accept sparse matrices.

        """
        # over-sampling method does not handle sparse matrix
        X, y = check_X_y(X, y)

        return super(BaseOverSampler, self).fit(X, y)

    def sample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like, shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : array-like, shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like, shape (n_samples_new,)
            The corresponding label of `X_resampled`

        Notes
        -----
        Over-samplers do not accept sparse matrices.

        """

        # Check the consistency of X and y
        X, y = check_X_y(X, y)

        return super(BaseOverSampler, self).sample(X, y)
