"""
Base class for the ensemble method.
"""
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import logging

from sklearn.utils import check_X_y

from ..base import SamplerMixin
from ..utils import check_ratio


class BaseEnsembleSampler(SamplerMixin):
    """Base class for ensemble algorithms.

    Warning: This class should not be used directly. Use the derive classes
    instead.
    """

    def __init__(self, ratio='auto', random_state=None):
        self.ratio = ratio
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

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
        X, y = check_X_y(X, y)
        self.ratio_ = check_ratio(self.ratio, y, 'under-sampling')

        return self
