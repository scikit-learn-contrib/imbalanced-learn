"""Class to perform under-sampling using easy ensemble."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numpy as np

from sklearn.utils import check_random_state

from .base import BaseEnsembleSampler
from ..under_sampling import RandomUnderSampler

MAX_INT = np.iinfo(np.int32).max


class EasyEnsemble(BaseEnsembleSampler):
    """Create an ensemble sets by iteratively applying random under-sampling.

    This method iteratively select a random subset and make an ensemble of the
    different sets.

    Read more in the :ref:`User Guide <ensemble_samplers>`.

    Parameters
    ----------
    ratio : str, dict, or callable, optional (default='auto')
        Ratio to use for resampling the data set.

        - If ``str``, has to be one of: (i) ``'minority'``: resample the
          minority class; (ii) ``'majority'``: resample the majority class,
          (iii) ``'not minority'``: resample all classes apart of the minority
          class, (iv) ``'all'``: resample all classes, and (v) ``'auto'``:
          correspond to ``'all'`` with for over-sampling methods and ``'not
          minority'`` for under-sampling methods. The classes targeted will be
          over-sampled or under-sampled to achieve an equal number of sample
          with the majority or minority class.
        - If ``dict``, the keys correspond to the targeted classes. The values
          correspond to the desired number of samples.
        - If callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples.

    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly
        selected from the majority class.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number
        generator; If ``RandomState`` instance, random_state is the random
        number generator; If ``None``, the random number generator is the
        ``RandomState`` instance used by ``np.random``.

    replacement : bool, optional (default=False)
        Whether or not to sample randomly with replacement or not.

    n_subsets : int, optional (default=10)
        Number of subsets to generate.

    Notes
    -----
    The method is described in [1]_.

    Supports mutli-class resampling by sampling each class independently.

    See :ref:`sphx_glr_auto_examples_ensemble_plot_easy_ensemble.py`.

    See also
    --------
    BalanceCascade, BalancedBaggingClassifier

    References
    ----------
    .. [1] X. Y. Liu, J. Wu and Z. H. Zhou, "Exploratory Undersampling for
       Class-Imbalance Learning," in IEEE Transactions on Systems, Man, and
       Cybernetics, Part B (Cybernetics), vol. 39, no. 2, pp. 539-550,
       April 2009.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.ensemble import \
EasyEnsemble # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> ee = EasyEnsemble(random_state=42)
    >>> X_res, y_res = ee.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res[0])))
    Resampled dataset shape Counter({0: 100, 1: 100})

    """

    def __init__(self,
                 ratio='auto',
                 return_indices=False,
                 random_state=None,
                 replacement=False,
                 n_subsets=10):
        super(EasyEnsemble, self).__init__(ratio=ratio,
                                           random_state=random_state)
        self.return_indices = return_indices
        self.replacement = replacement
        self.n_subsets = n_subsets

    def _sample(self, X, y):
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

        idx_under : ndarray, shape (n_subset, n_samples, )
            If `return_indices` is `True`, a boolean array will be returned
            containing the which samples have been selected.

        """

        random_state = check_random_state(self.random_state)

        X_resampled = []
        y_resampled = []
        if self.return_indices:
            idx_under = []

        for _ in range(self.n_subsets):
            rus = RandomUnderSampler(
                ratio=self.ratio_, return_indices=True,
                random_state=random_state.randint(MAX_INT),
                replacement=self.replacement)
            sel_x, sel_y, sel_idx = rus.fit_sample(X, y)
            X_resampled.append(sel_x)
            y_resampled.append(sel_y)
            if self.return_indices:
                idx_under.append(sel_idx)

        if self.return_indices:
            return (np.array(X_resampled), np.array(y_resampled),
                    np.array(idx_under))
        else:
            return np.array(X_resampled), np.array(y_resampled)
