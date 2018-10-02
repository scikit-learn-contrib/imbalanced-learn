"""Transform a dataset into an imbalanced dataset."""

# Authors: Dayvid Oliveira
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import warnings
from collections import Counter

from sklearn.utils import check_X_y

from ..under_sampling import RandomUnderSampler
from ..utils import check_sampling_strategy


def make_imbalance(X,
                   y,
                   sampling_strategy=None,
                   ratio=None,
                   random_state=None,
                   verbose=False,
                   **kwargs):
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

    sampling_strategy : dict, or callable,
        Ratio to use for resampling the data set.

        - When ``dict``, the keys correspond to the targeted classes. The
          values correspond to the desired number of samples for each targeted
          class.

        - When callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples for each class.

    ratio : str, dict, or callable
        .. deprecated:: 0.4
           Use the parameter ``sampling_strategy`` instead. It will be removed
           in 0.6.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random.

    verbose : bool, optional (default=False)
        Show information regarding the sampling.

    kwargs : dict, optional
        Dictionary of additional keyword arguments to pass to
        ``sampling_strategy``.

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
    :ref:`sphx_glr_auto_examples_plot_sampling_strategy_usage.py`.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import load_iris
    >>> from imblearn.datasets import make_imbalance

    >>> data = load_iris()
    >>> X, y = data.data, data.target
    >>> print('Distribution before imbalancing: {}'.format(Counter(y)))
    Distribution before imbalancing: Counter({0: 50, 1: 50, 2: 50})
    >>> X_res, y_res = make_imbalance(X, y,
    ...                               sampling_strategy={0: 10, 1: 20, 2: 30},
    ...                               random_state=42)
    >>> print('Distribution after imbalancing: {}'.format(Counter(y_res)))
    Distribution after imbalancing: Counter({2: 30, 1: 20, 0: 10})

    """
    X, y = check_X_y(X, y)
    target_stats = Counter(y)
    # restrict ratio to be a dict or a callable
    # FIXME remove ratio at 0.6
    if ratio is not None:
        warnings.warn("'ratio' has been deprecated in 0.4 and will be "
                      "removed in 0.6. Use 'sampling_strategy' instead.")
        sampling_strategy = ratio
    elif sampling_strategy is None:
        raise TypeError("make_imbalance() missing 1 required positional "
                        "argument: 'sampling_strategy'")
    if isinstance(sampling_strategy, dict) or callable(sampling_strategy):
        sampling_strategy_ = check_sampling_strategy(
            sampling_strategy, y, 'under-sampling', **kwargs)
    else:
        raise ValueError("'sampling_strategy' has to be a dictionary or a "
                         "function returning a dictionary. Got {} instead."
                         .format(type(sampling_strategy)))

    if verbose:
        print('The original target distribution in the dataset is: %s',
              target_stats)
    rus = RandomUnderSampler(
        sampling_strategy=sampling_strategy_,
        replacement=False,
        random_state=random_state)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    if verbose:
        print('Make the dataset imbalanced: %s', Counter(y_resampled))

    return X_resampled, y_resampled
