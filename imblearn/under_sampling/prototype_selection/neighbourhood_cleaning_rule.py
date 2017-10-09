"""Class performing under-sampling based on the neighbourhood cleaning rule."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import division, print_function

from collections import Counter

import numpy as np
from scipy.stats import mode

from sklearn.utils import safe_indexing

from ..base import BaseCleaningSampler
from .edited_nearest_neighbours import EditedNearestNeighbours
from ...utils import check_neighbors_object, check_ratio

SEL_KIND = ('all', 'mode')


class NeighbourhoodCleaningRule(BaseCleaningSampler):
    """Class performing under-sampling based on the neighbourhood cleaning
    rule.

    Read more in the :ref:`User Guide <condensed_nearest_neighbors>`.

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

        .. warning::
           This algorithm is a cleaning under-sampling method. When providing a
           ``dict``, only the targeted classes will be used; the number of
           samples will be discarded.

    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly
        selected from the majority class.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number
        generator; If ``RandomState`` instance, random_state is the random
        number generator; If ``None``, the random number generator is the
        ``RandomState`` instance used by ``np.random``.

    size_ngh : int, optional (default=None)
        Size of the neighbourhood to consider to compute the nearest-neighbors.

        .. deprecated:: 0.2
           ``size_ngh`` is deprecated from 0.2 and will be replaced in 0.4
           Use ``n_neighbors`` instead.

    n_neighbors : int or object, optional (default=3)
        If ``int``, size of the neighbourhood to consider to compute the
        nearest neighbors. If object, an estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the nearest-neighbors.

    threshold_cleaning : float, optional (default=0.5)
        Threshold used to whether consider a class or not during the cleaning
        after applying ENN. A class will be considered during cleaning when:

        Ci > C x T ,

        where Ci and C is the number of samples in the class and the data set,
        respectively and theta is the threshold.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    Notes
    -----
    See the original paper: [1]_.

    Supports mutli-class resampling. A one-vs.-rest scheme is used when
    sampling a class as proposed in [1]_.

    See
    :ref:`sphx_glr_auto_examples_under-sampling_plot_neighbourhood_cleaning_rule.py`.

    References
    ----------
    .. [1] J. Laurikkala, "Improving identification of difficult small classes
       by balancing class distribution," Springer Berlin Heidelberg, 2001.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import \
NeighbourhoodCleaningRule # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> ncr = NeighbourhoodCleaningRule(random_state=42)
    >>> X_res, y_res = ncr.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({1: 877, 0: 100})

    """

    def __init__(self,
                 ratio='auto',
                 return_indices=False,
                 random_state=None,
                 size_ngh=None,
                 n_neighbors=3,
                 kind_sel='all',
                 threshold_cleaning=0.5,
                 n_jobs=1):
        super(NeighbourhoodCleaningRule, self).__init__(
            ratio=ratio, random_state=random_state)
        self.return_indices = return_indices
        self.size_ngh = size_ngh
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.threshold_cleaning = threshold_cleaning
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Create the objects required by NCR."""
        # FIXME: Deprecated from 0.2. To be removed in 0.4.
        self.nn_ = check_neighbors_object('n_neighbors', self.n_neighbors,
                                          additional_neighbor=1)
        self.nn_.set_params(**{'n_jobs': self.n_jobs})

        if self.kind_sel not in SEL_KIND:
            raise NotImplementedError

        if self.threshold_cleaning > 1 or self.threshold_cleaning < 0:
            raise ValueError("'threshold_cleaning' is a value between 0 and 1."
                             " Got {} instead.".format(
                                 self.threshold_cleaning))

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
            If `return_indices` is `True`, a boolean array will be returned
            containing the which samples have been selected.

        """
        self._validate_estimator()
        enn = EditedNearestNeighbours(ratio=self.ratio, return_indices=True,
                                      random_state=self.random_state,
                                      size_ngh=self.size_ngh,
                                      n_neighbors=self.n_neighbors,
                                      kind_sel='mode',
                                      n_jobs=self.n_jobs)
        _, _, index_not_a1 = enn.fit_sample(X, y)
        index_a1 = np.ones(y.shape, dtype=bool)
        index_a1[index_not_a1] = False
        index_a1 = np.flatnonzero(index_a1)

        # clean the neighborhood
        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)
        # compute which classes to consider for cleaning for the A2 group
        classes_under_sample = [c for c, n_samples in target_stats.items()
                                if (c in self.ratio_.keys() and
                                    (n_samples > X.shape[0] *
                                     self.threshold_cleaning))]
        self.nn_.fit(X)
        class_minority_indices = np.flatnonzero(y == class_minority)
        X_class = safe_indexing(X, class_minority_indices)
        y_class = safe_indexing(y, class_minority_indices)
        nnhood_idx = self.nn_.kneighbors(
            X_class, return_distance=False)[:, 1:]
        nnhood_label = y[nnhood_idx]
        if self.kind_sel == 'mode':
            nnhood_label_majority, _ = mode(nnhood_label, axis=1)
            nnhood_bool = np.ravel(nnhood_label_majority) == y_class
        elif self.kind_sel == 'all':
            nnhood_label_majority = nnhood_label == class_minority
            nnhood_bool = np.all(nnhood_label, axis=1)
        else:
            raise NotImplementedError
        # compute a2 group
        index_a2 = np.ravel(nnhood_idx[~nnhood_bool])
        index_a2 = np.unique([index for index in index_a2
                              if y[index] in classes_under_sample])

        union_a1_a2 = np.union1d(index_a1, index_a2).astype(int)
        selected_samples = np.ones(y.shape, dtype=bool)
        selected_samples[union_a1_a2] = False
        index_target_class = np.flatnonzero(selected_samples)

        if self.return_indices:
            return (safe_indexing(X, index_target_class),
                    safe_indexing(y, index_target_class),
                    index_target_class)
        else:
            return (safe_indexing(X, index_target_class),
                    safe_indexing(y, index_target_class))
