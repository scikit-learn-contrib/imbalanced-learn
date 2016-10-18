"""Transform a dataset into an imbalanced dataset."""

import logging
from collections import Counter

import numpy as np
from sklearn.utils import check_random_state, check_X_y

LOGGER = logging.getLogger(__name__)


def make_imbalance(X, y, ratio, min_c_=None, random_state=None):
    """Turns a dataset into an imbalanced dataset at specific ratio.
    A simple toy dataset to visualize clustering and classification
    algorithms.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Matrix containing the data to be imbalanced.

    y : ndarray, shape (n_samples, )
        Corresponding label for each sample in X.

    ratio : float,
        The desired ratio given by the number of samples in
        the minority class over the the number of samples in
        the majority class. Thus the ratio should be in the interval [0., 1.]

    min_c_ : str or int, optional (default=None)
        The identifier of the class to be the minority class.
        If None, min_c_ is set to be the current minority class.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    Returns
    -------
    X_resampled : ndarray, shape (n_samples_new, n_features)
        The array containing the imbalanced data.

    y_resampled : ndarray, shape (n_samples_new)
        The corresponding label of `X_resampled`

    """
    if isinstance(ratio, float):
        if ratio > 1:
            raise ValueError('Ration cannot be greater than one.')
        elif ratio <= 0:
            raise ValueError('Ratio cannot be negative.')
    else:
        raise ValueError('Ratio must be a float between 0.0 < ratio < 1.0')

    X, y = check_X_y(X, y)

    random_state = check_random_state(random_state)

    stats_c_ = Counter(y)

    LOGGER.info('The original target distribution in the dataset is: %s',
                stats_c_)

    if min_c_ is None:
        min_c_ = min(stats_c_, key=stats_c_.get)

    n_min_samples = int(np.count_nonzero(y != min_c_) * ratio)
    if n_min_samples > stats_c_[min_c_]:
        raise ValueError('Current imbalance ratio of data is lower than'
                         ' desired ratio!')
    if n_min_samples == 0:
        raise ValueError('Not enough samples for desired ratio!')

    mask = y == min_c_

    idx_maj = np.where(~mask)[0]
    idx_min = np.where(mask)[0]
    idx_min = random_state.choice(idx_min, size=n_min_samples, replace=False)
    idx = np.concatenate((idx_min, idx_maj), axis=0)

    X_resampled, y_resampled = X[idx, :], y[idx]

    LOGGER.info('Make the dataset imbalanced: %s', Counter(y_resampled))

    return X_resampled, y_resampled
