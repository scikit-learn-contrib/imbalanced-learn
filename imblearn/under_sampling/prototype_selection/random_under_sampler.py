"""Class to perform random under-sampling."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import division

import numpy as np
from sklearn.utils import check_random_state, safe_indexing

from ..base import BaseUnderSampler


class RandomUnderSampler(BaseUnderSampler):
    """Class to perform random under-sampling.

    Under-sample the majority class(es) by randomly picking samples
    with or without replacement.

    Read more in the :ref:`User Guide <controlled_under_sampling>`.

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
        Whether or not to return the indices of the samples randomly selected
        from the majority class.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number
        generator; If ``RandomState`` instance, random_state is the random
        number generator; If ``None``, the random number generator is the
        ``RandomState`` instance used by ``np.random``.

    replacement : boolean, optional (default=False)
        Whether the sample is with or without replacement.

    Notes
    -----
    Supports mutli-class resampling by sampling each class independently.

    See
    :ref:`sphx_glr_auto_examples_plot_ratio_usage.py` and
    :ref:`sphx_glr_auto_examples_under-sampling_plot_random_under_sampler.py`

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import \
RandomUnderSampler # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ...  weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> rus = RandomUnderSampler(random_state=42)
    >>> X_res, y_res = rus.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({0: 100, 1: 100})

    """

    def __init__(self,
                 ratio='auto',
                 return_indices=False,
                 random_state=None,
                 replacement=False):
        super(RandomUnderSampler, self).__init__(
            ratio=ratio, random_state=random_state)
        self.return_indices = return_indices
        self.replacement = replacement

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
(n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new,)
            The corresponding label of `X_resampled`

        idx_under : ndarray, shape (n_samples, )
            If `return_indices` is `True`, an array will be returned
            containing a boolean for each sample to represent whether
            that sample was selected or not.

        """
        random_state = check_random_state(self.random_state)

        idx_under = np.empty((0, ), dtype=int)

        for target_class in np.unique(y):
            if target_class in self.ratio_.keys():
                n_samples = self.ratio_[target_class]
                index_target_class = random_state.choice(
                    range(np.count_nonzero(y == target_class)),
                    size=n_samples,
                    replace=self.replacement)
            else:
                index_target_class = slice(None)

            idx_under = np.concatenate(
                (idx_under, np.flatnonzero(y == target_class)[
                    index_target_class]), axis=0)

        if self.return_indices:
            return (safe_indexing(X, idx_under), safe_indexing(y, idx_under),
                    idx_under)
        else:
            return safe_indexing(X, idx_under), safe_indexing(y, idx_under)
