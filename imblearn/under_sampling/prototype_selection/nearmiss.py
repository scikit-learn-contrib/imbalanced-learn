"""Class to perform under-sampling based on nearmiss methods."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import division

import warnings
from collections import Counter

import numpy as np

from ..base import BaseUnderSampler
from ...utils import check_neighbors_object
from ...utils.deprecation import deprecate_parameter


class NearMiss(BaseUnderSampler):
    """Class to perform under-sampling based on NearMiss methods.

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

    version : int, optional (default=1)
        Version of the NearMiss to use. Possible values are 1, 2 or 3.

    size_ngh : int, optional (default=None)
        Size of the neighbourhood to consider to compute the average
        distance to the minority point samples.

        .. deprecated:: 0.2
           ``size_ngh`` is deprecated from 0.2 and will be replaced in 0.4
           Use ``n_neighbors`` instead.

    n_neighbors : int or object, optional (default=3)
        If ``int``, size of the neighbourhood to consider to compute the
        average distance to the minority point samples.  If object, an
        estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.

    ver3_samp_ngh : int, optional (default=3)
        NearMiss-3 algorithm start by a phase of re-sampling. This
        parameter correspond to the number of neighbours selected
        create the sub_set in which the selection will be performed.

        .. deprecated:: 0.2
           ``ver3_samp_ngh`` is deprecated from 0.2 and will be replaced
           in 0.4. Use ``n_neighbors_ver3`` instead.

    n_neighbors_ver3 : int or object, optional (default=3)
        If ``int``, NearMiss-3 algorithm start by a phase of re-sampling. This
        parameter correspond to the number of neighbours selected create the
        subset in which the selection will be performed.  If object, an
        estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    Notes
    -----
    The methods are based on [1]_.

    Supports mutli-class resampling.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import \
NearMiss # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> nm = NearMiss(random_state=42)
    >>> X_res, y_res = nm.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({0: 100, 1: 100})

    References
    ----------
    .. [1] I. Mani, I. Zhang. "kNN approach to unbalanced data distributions:
       a case study involving information extraction," In Proceedings of
       workshop on learning from imbalanced datasets, 2003.

    """

    def __init__(self,
                 ratio='auto',
                 return_indices=False,
                 random_state=None,
                 version=1,
                 size_ngh=None,
                 n_neighbors=3,
                 ver3_samp_ngh=None,
                 n_neighbors_ver3=3,
                 n_jobs=1):
        super(NearMiss, self).__init__(ratio=ratio, random_state=random_state)
        self.return_indices = return_indices
        self.version = version
        self.size_ngh = size_ngh
        self.n_neighbors = n_neighbors
        self.ver3_samp_ngh = ver3_samp_ngh
        self.n_neighbors_ver3 = n_neighbors_ver3
        self.n_jobs = n_jobs

    def _selection_dist_based(self,
                              X,
                              y,
                              dist_vec,
                              num_samples,
                              key,
                              sel_strategy='nearest'):
        """Select the appropriate samples depending of the strategy selected.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Original samples.

        y : ndarray, shape (n_samples, )
            Associated label to X.

        dist_vec : ndarray, shape (n_samples, )
            The distance matrix to the nearest neigbour.

        num_samples: int
            The desired number of samples to select.

        key : str or int,
            The target class.

        sel_strategy : str, optional (default='nearest')
            Strategy to select the samples. Either 'nearest' or 'farthest'

        Returns
        -------
        X_sel : ndarray, shape (num_samples, n_features)
            Selected samples.

        y_sel : ndarray, shape (num_samples, )
            The associated label.

        idx_sel : ndarray, shape (num_samples, )
            The list of the indices of the selected samples.

        """

        # Compute the distance considering the farthest neighbour
        dist_avg_vec = np.sum(dist_vec[:, -self.nn_.n_neighbors:], axis=1)

        if dist_vec.shape[0] != X[y == key].shape[0]:
            raise RuntimeError('The samples to be selected do not correspond'
                               ' to the distance matrix given. Ensure that'
                               ' both `X[y == key]` and `dist_vec` are'
                               ' related.')

        # Sort the list of distance and get the index
        if sel_strategy == 'nearest':
            sort_way = False
        elif sel_strategy == 'farthest':
            sort_way = True
        else:
            raise NotImplementedError

        sorted_idx = sorted(
            range(len(dist_avg_vec)),
            key=dist_avg_vec.__getitem__,
            reverse=sort_way)

        # Throw a warning to tell the user that we did not have enough samples
        # to select and that we just select everything
        if len(sorted_idx) < num_samples:
            warnings.warn('The number of the samples to be selected is larger'
                          ' than the number of samples available. The'
                          ' balancing ratio cannot be ensure and all samples'
                          ' will be returned.')

        # Select the desired number of samples
        return sorted_idx[:num_samples]

    def _validate_estimator(self):
        """Private function to create the NN estimator"""
        # FIXME: Deprecated in 0.2. To be removed in 0.4.
        deprecate_parameter(self, '0.2', 'size_ngh', 'n_neighbors')
        if self.version == 3:
            deprecate_parameter(self, '0.2', 'ver3_samp_ngh',
                                'n_neighbors_ver3')

        self.nn_ = check_neighbors_object('n_neighbors', self.n_neighbors)
        self.nn_.set_params(**{'n_jobs': self.n_jobs})

        if self.version == 3:
            self.nn_ver3_ = check_neighbors_object('n_neighbors_ver3',
                                                   self.n_neighbors_ver3)
            self.nn_ver3_.set_params(**{'n_jobs': self.n_jobs})

        if self.version not in (1, 2, 3):
            raise ValueError('Parameter `version` must be 1, 2 or 3, got'
                             ' {}'.format(self.version))

    def _sample(self, X, y):
        """Resample the dataset.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : ndarray, shape (n_samples, )
            Corresponding label for each sample in X.

        Returns
        -------
        X_resampled : ndarray, shape (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_samples_new)
            The corresponding label of `X_resampled`

        idx_under : ndarray, shape (n_samples, )
            If `return_indices` is `True`, a boolean array will be returned
            containing the which samples have been selected.

        """
        self._validate_estimator()

        X_resampled = np.empty((0, X.shape[1]), dtype=X.dtype)
        y_resampled = np.empty((0, ), dtype=y.dtype)
        if self.return_indices:
            idx_under = np.empty((0, ), dtype=int)

        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)

        self.nn_.fit(X[y == class_minority])

        for target_class in np.unique(y):
            if target_class in self.ratio_.keys():
                n_samples = self.ratio_[target_class]
                X_class = X[y == target_class]
                y_class = y[y == target_class]

                if self.version == 1:
                    dist_vec, idx_vec = self.nn_.kneighbors(
                        X_class, n_neighbors=self.nn_.n_neighbors)
                    index_target_class = self._selection_dist_based(
                        X, y, dist_vec, n_samples, target_class,
                        sel_strategy='nearest')
                elif self.version == 2:
                    dist_vec, idx_vec = self.nn_.kneighbors(
                        X_class, n_neighbors=target_stats[class_minority])
                    index_target_class = self._selection_dist_based(
                        X, y, dist_vec, n_samples, target_class,
                        sel_strategy='nearest')
                elif self.version == 3:
                    self.nn_ver3_.fit(X_class)
                    dist_vec, idx_vec = self.nn_ver3_.kneighbors(
                        X[y == class_minority])
                    idx_vec_farthest = np.unique(idx_vec.reshape(-1))
                    X_class_selected = X_class[idx_vec_farthest, :]
                    y_class_selected = y_class[idx_vec_farthest]

                    dist_vec, idx_vec = self.nn_.kneighbors(
                        X_class_selected, n_neighbors=self.nn_.n_neighbors)
                    index_target_class = self._selection_dist_based(
                        X_class_selected, y_class_selected, dist_vec,
                        n_samples, target_class, sel_strategy='farthest')
                    # idx_tmp is relative to the feature selected in the
                    # previous step and we need to find the indirection
                    index_target_class = idx_vec_farthest[index_target_class]
            else:
                index_target_class = slice(None)

            X_resampled = np.concatenate(
                (X_resampled, X[y == target_class][index_target_class]),
                axis=0)
            y_resampled = np.concatenate(
                (y_resampled, y[y == target_class][index_target_class]),
                axis=0)
            if self.return_indices:
                idx_under = np.concatenate(
                    (idx_under, np.flatnonzero(y == target_class)[
                        index_target_class]), axis=0)

        if self.return_indices:
            return X_resampled, y_resampled, idx_under
        else:
            return X_resampled, y_resampled
