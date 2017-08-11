"""Transform a dataset into an imbalanced dataset."""


# Authors: Dayvid Oliveira
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import logging
import warnings
from collections import Counter
from numbers import Real

from sklearn.utils import check_X_y

from ..under_sampling.prototype_selection import RandomUnderSampler
from ..utils import check_ratio

LOGGER = logging.getLogger(__name__)


def make_imbalance(X, y, ratio, min_c_=None, random_state=None, **kwargs):
    """Turns a dataset into an imbalanced dataset at specific ratio.

    A simple toy dataset to visualize clustering and classification
    algorithms.

    Read more in the :ref:`User Guide <make_imbalanced>`.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Matrix containing the data to be imbalanced.

    y : ndarray, shape (n_samples, )
        Corresponding label for each sample in X.

    ratio : str, dict, or callable, optional (default='auto')
        Ratio to use for resampling the data set.

        - If ``dict``, the keys correspond to the targeted classes. The values
          correspond to the desired number of samples. All samples will be
          passed through if the class is not specified.
        - If callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples.

    min_c_ : str or int, optional (default=None)
        The identifier of the class to be the minority class.
        If ``None``, ``min_c_`` is set to be the current minority class.
        Only used when ``ratio`` is a float for back-compatibility.

        .. deprecated:: 0.2
           ``min_c_`` is deprecated in 0.2 and will be removed in 0.4. Use
           ``ratio`` by passing a ``dict`` instead.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    kwargs : dict, optional
        Dictionary of additional keyword arguments to pass to ``ratio``.

    Returns
    -------
    X_resampled : ndarray, shape (n_samples_new, n_features)
        The array containing the imbalanced data.

    y_resampled : ndarray, shape (n_samples_new)
        The corresponding label of `X_resampled`

    Notes
    -----
    See
    :ref:`sphx_glr_auto_examples_applications_plot_multi_class_under_sampling.py`,
    :ref:`sphx_glr_auto_examples_datasets_plot_make_imbalance.py`, and
    :ref:`sphx_glr_auto_examples_plot_ratio_usage.py`.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import load_iris
    >>> from imblearn.datasets import make_imbalance

    >>> data = load_iris()
    >>> X, y = data.data, data.target
    >>> print('Distribution before imbalancing: {}'.format(Counter(y)))
    Distribution before imbalancing: Counter({0: 50, 1: 50, 2: 50})
    >>> X_res, y_res = make_imbalance(X, y, ratio={0: 10, 1: 20, 2: 30},
    ...                               random_state=42)
    >>> print('Distribution after imbalancing: {}'.format(Counter(y_res)))
    Distribution after imbalancing: Counter({2: 30, 1: 20, 0: 10})

    """
    X, y = check_X_y(X, y)
    target_stats = Counter(y)
    # restrict ratio to be a dict or a callable
    if isinstance(ratio, dict) or callable(ratio):
        ratio_ = check_ratio(ratio, y, 'under-sampling', **kwargs)
    # FIXME: deprecated in 0.2 to be removed in 0.4
    elif isinstance(ratio, Real):
        if min_c_ is None:
            min_c_ = min(target_stats, key=target_stats.get)
        else:
            warnings.warn("'min_c_' is deprecated in 0.2 and will be removed"
                          " in 0.4. Use 'ratio' as dictionary instead.",
                          DeprecationWarning)
        warnings.warn("'ratio' being a float is deprecated in 0.2 and will not"
                      " be supported in 0.4. Use a dictionary instead.",
                      DeprecationWarning)
        class_majority = max(target_stats, key=target_stats.get)
        ratio_ = {}
        for class_sample, n_sample in target_stats.items():
            if class_sample == min_c_:
                n_min_samples = int(target_stats[class_majority] * ratio)
                ratio_[class_sample] = n_min_samples
            else:
                ratio_[class_sample] = n_sample
        ratio_ = check_ratio(ratio_, y, 'under-sampling')
    else:
        raise ValueError("'ratio' has to be a dictionary or a function"
                         " returning a dictionary. Got {} instead.".format(
                             type(ratio)))

    LOGGER.info('The original target distribution in the dataset is: %s',
                target_stats)
    rus = RandomUnderSampler(ratio=ratio_, replacement=False,
                             random_state=random_state)
    X_resampled, y_resampled = rus.fit_sample(X, y)
    LOGGER.info('Make the dataset imbalanced: %s', Counter(y_resampled))

    return X_resampled, y_resampled
