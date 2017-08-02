"""Transform a dataset into an imbalanced dataset."""


# Authors: Dayvid Oliveira
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import logging
import warnings
from collections import Counter
from numbers import Real

import numpy as np
from sklearn.utils import check_X_y

from ..under_sampling.prototype_selection import RandomUnderSampler
from ..utils import check_ratio

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

    ratio : str, dict, or callable, optional (default='auto')
        Ratio to use for resampling the data set.

        - If ``dict``, the keys correspond to the targeted classes. The values
          correspond to the desired number of samples.
        - If callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples.

    min_c_ : str or int, optional (default=None)
        The identifier of the class to be the minority class.
        If None, min_c_ is set to be the current minority class.

        .. deprecated:: 0.2
           ``min_c_`` is deprecated in 0.2 and will be removed in 0.4. Use
           ``ratio`` by passing a ``dict`` instead.

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
    X, y = check_X_y(X, y)
    stats_c_ = Counter(y)
    # restrict ratio to be a dict or a callable
    if isinstance(ratio, dict) or callable(ratio):
        ratio = check_ratio(ratio, y, 'under-sampling')
    # FIXME: deprecated in 0.2 to be removed in 0.4
    elif isinstance(ratio, Real):
        if min_c_ is None:
            min_c_ = min(stats_c_, key=stats_c_.get)
        else:
            warnings.warn("'min_c_' is deprecated in 0.2 and will be removed"
                          " in 0.4. Use 'ratio' as dictionary instead.",
                          DeprecationWarning)
        warnings.warn("'ratio' being a float is deprecated in 0.2 and will not"
                      " be supported in 0.4. Use a dictionary instead.",
                      DeprecationWarning)
        class_majority = max(stats_c_, key=stats_c_.get)
        ratio = {}
        for label, n_sample in stats_c_.items():
            if label == min_c_:
                n_min_samples = int(stats_c_[class_majority] * ratio)
                ratio[label] = n_min_samples
            else:
                ratio[label] = n_sample
        ratio = check_ratio(ratio, y, 'under-sampling')
    else:
        raise ValueError("'ratio' has to be a dictionary or a function"
                         " returning a dictionary. Got {} instead.".format(
                             type(ratio)))

    LOGGER.info('The original target distribution in the dataset is: %s',
                stats_c_)
    X_resampled, y_resampled = RandomUnderSampler(ratio=ratio,
                                                  replacement=False,
                                                  random_state=random_state)
    LOGGER.info('Make the dataset imbalanced: %s', Counter(y_resampled))

    return X_resampled, y_resampled
