"""Class performing under-sampling based on the neighbourhood cleaning rule."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import division

from collections import Counter

import numpy as np
from scipy.stats import mode

from sklearn.utils import safe_indexing

from ..base import BaseCleaningSampler
from ._edited_nearest_neighbours import EditedNearestNeighbours
from ...utils import check_neighbors_object
from ...utils import Substitution
from ...utils.deprecation import deprecate_parameter
from ...utils._docstring import _random_state_docstring

SEL_KIND = ('all', 'mode')


@Substitution(
    sampling_strategy=BaseCleaningSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring)
class NeighbourhoodCleaningRule(BaseCleaningSampler):
    """Class performing under-sampling based on the neighbourhood cleaning
    rule.

    Read more in the :ref:`User Guide <condensed_nearest_neighbors>`.

    Parameters
    ----------
    {sampling_strategy}

    return_indices : bool, optional (default=False)
        Whether or not to return the indices of the samples randomly
        selected.

        .. deprecated:: 0.4
           ``return_indices`` is deprecated. Use the attribute
           ``sample_indices_`` instead.

    {random_state}

        .. deprecated:: 0.4
           ``random_state`` is deprecated in 0.4 and will be removed in 0.6.

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

    ratio : str, dict, or callable
        .. deprecated:: 0.4
           Use the parameter ``sampling_strategy`` instead. It will be removed
           in 0.6.

    Attributes
    ----------
    sample_indices_ : ndarray, shape (n_new_samples)
        Indices of the samples selected.

        .. versionadded:: 0.4
           ``sample_indices_`` used instead of ``return_indices=True``.

    Notes
    -----
    See the original paper: [1]_.

    Supports multi-class resampling. A one-vs.-rest scheme is used when
    sampling a class as proposed in [1]_.

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
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> ncr = NeighbourhoodCleaningRule()
    >>> X_res, y_res = ncr.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{1: 877, 0: 100}})

    """

    def __init__(self,
                 sampling_strategy='auto',
                 return_indices=False,
                 random_state=None,
                 n_neighbors=3,
                 kind_sel='all',
                 threshold_cleaning=0.5,
                 n_jobs=1,
                 ratio=None):
        super(NeighbourhoodCleaningRule, self).__init__(
            sampling_strategy=sampling_strategy, ratio=ratio)
        self.random_state = random_state
        self.return_indices = return_indices
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.threshold_cleaning = threshold_cleaning
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Create the objects required by NCR."""

        # check for deprecated random_state
        if self.random_state is not None:
            deprecate_parameter(self, '0.4', 'random_state')

        self.nn_ = check_neighbors_object(
            'n_neighbors', self.n_neighbors, additional_neighbor=1)
        self.nn_.set_params(**{'n_jobs': self.n_jobs})

        if self.kind_sel not in SEL_KIND:
            raise NotImplementedError

        if self.threshold_cleaning > 1 or self.threshold_cleaning < 0:
            raise ValueError(
                "'threshold_cleaning' is a value between 0 and 1."
                " Got {} instead.".format(self.threshold_cleaning))

    def _fit_resample(self, X, y):
        if self.return_indices:
            deprecate_parameter(self, '0.4', 'return_indices',
                                'sample_indices_')
        self._validate_estimator()
        enn = EditedNearestNeighbours(
            sampling_strategy=self.sampling_strategy,
            n_neighbors=self.n_neighbors,
            kind_sel='mode',
            n_jobs=self.n_jobs,
            ratio=self.ratio)
        enn.fit_resample(X, y)
        index_not_a1 = enn.sample_indices_
        index_a1 = np.ones(y.shape, dtype=bool)
        index_a1[index_not_a1] = False
        index_a1 = np.flatnonzero(index_a1)

        # clean the neighborhood
        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)
        # compute which classes to consider for cleaning for the A2 group
        classes_under_sample = [
            c for c, n_samples in target_stats.items()
            if (c in self.sampling_strategy_.keys() and (
                n_samples > X.shape[0] * self.threshold_cleaning))
        ]
        self.nn_.fit(X)
        class_minority_indices = np.flatnonzero(y == class_minority)
        X_class = safe_indexing(X, class_minority_indices)
        y_class = safe_indexing(y, class_minority_indices)
        nnhood_idx = self.nn_.kneighbors(X_class, return_distance=False)[:, 1:]
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
        index_a2 = np.unique(
            [index for index in index_a2 if y[index] in classes_under_sample])

        union_a1_a2 = np.union1d(index_a1, index_a2).astype(int)
        selected_samples = np.ones(y.shape, dtype=bool)
        selected_samples[union_a1_a2] = False
        self.sample_indices_ = np.flatnonzero(selected_samples)

        if self.return_indices:
            return (safe_indexing(X, self.sample_indices_),
                    safe_indexing(y, self.sample_indices_), self.sample_indices_)
        return (safe_indexing(X, self.sample_indices_),
                safe_indexing(y, self.sample_indices_))
