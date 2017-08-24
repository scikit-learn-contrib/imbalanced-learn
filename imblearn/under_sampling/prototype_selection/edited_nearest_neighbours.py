"""Class to perform under-sampling based on the edited nearest neighbour
method."""


# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Dayvid Oliveira
#          Christos Aridas
# License: MIT

from __future__ import division

from collections import Counter

import numpy as np
from scipy.stats import mode

from sklearn.utils import safe_indexing

from ..base import BaseCleaningSampler
from ...utils import check_neighbors_object
from ...utils.deprecation import deprecate_parameter

SEL_KIND = ('all', 'mode')


class EditedNearestNeighbours(BaseCleaningSampler):
    """Class to perform under-sampling based on the edited nearest neighbour
    method.

    Read more in the :ref:`User Guide <edited_nearest_neighbors>`.

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

    kind_sel : str, optional (default='all')
        Strategy to use in order to exclude samples.

        - If ``'all'``, all neighbours will have to agree with the samples of
          interest to not be excluded.
        - If ``'mode'``, the majority vote of the neighbours will be used in
          order to exclude a sample.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    Notes
    -----
    The method is based on [1]_.

    Supports mutli-class resampling. A one-vs.-rest scheme is used when
    sampling a class as proposed in [1]_.

    See
    :ref:`sphx_glr_auto_examples_pipeline_plot_pipeline_classification.py` and
    :ref:`sphx_glr_auto_examples_under-sampling_plot_enn_renn_allknn.py`.

    See also
    --------
    CondensedNearestNeighbour, RepeatedEditedNearestNeighbours, AllKNN

    References
    ----------
    .. [1] D. Wilson, Asymptotic" Properties of Nearest Neighbor Rules Using
       Edited Data," In IEEE Transactions on Systems, Man, and Cybernetrics,
       vol. 2 (3), pp. 408-421, 1972.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import \
EditedNearestNeighbours # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> enn = EditedNearestNeighbours(random_state=42)
    >>> X_res, y_res = enn.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({1: 887, 0: 100})

    """

    def __init__(self,
                 ratio='auto',
                 return_indices=False,
                 random_state=None,
                 size_ngh=None,
                 n_neighbors=3,
                 kind_sel='all',
                 n_jobs=1):
        super(EditedNearestNeighbours, self).__init__(
            ratio=ratio,
            random_state=random_state)
        self.return_indices = return_indices
        self.size_ngh = size_ngh
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Validate the estimator created in the ENN."""
        # FIXME: Deprecated in 0.2. To be removed in 0.4
        deprecate_parameter(self, '0.2', 'size_ngh', 'n_neighbors')

        self.nn_ = check_neighbors_object('n_neighbors', self.n_neighbors,
                                          additional_neighbor=1)
        self.nn_.set_params(**{'n_jobs': self.n_jobs})

        if self.kind_sel not in SEL_KIND:
            raise NotImplementedError

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

        idx_under = np.empty((0, ), dtype=int)

        self.nn_.fit(X)

        for target_class in np.unique(y):
            if target_class in self.ratio_.keys():
                target_class_indices = np.flatnonzero(y == target_class)
                X_class = safe_indexing(X, target_class_indices)
                y_class = safe_indexing(y, target_class_indices)
                nnhood_idx = self.nn_.kneighbors(
                    X_class, return_distance=False)[:, 1:]
                nnhood_label = y[nnhood_idx]
                if self.kind_sel == 'mode':
                    nnhood_label, _ = mode(nnhood_label, axis=1)
                    nnhood_bool = np.ravel(nnhood_label) == y_class
                elif self.kind_sel == 'all':
                    nnhood_label = nnhood_label == target_class
                    nnhood_bool = np.all(nnhood_label, axis=1)
                index_target_class = np.flatnonzero(nnhood_bool)
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


class RepeatedEditedNearestNeighbours(BaseCleaningSampler):
    """Class to perform under-sampling based on the repeated edited nearest
    neighbour method.

    Read more in the :ref:`User Guide <edited_nearest_neighbors>`.

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

        .. deprecated: 0.2
           ``size_ngh`` is deprecated from 0.2 and will be replaced in 0.4
           Use ``n_neighbors`` instead.

    n_neighbors : int or object, optional (default=3)
        If ``int``, size of the neighbourhood to consider to compute the
        nearest neighbors. If object, an estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the nearest-neighbors.

    max_iter : int, optional (default=100)
        Maximum number of iterations of the edited nearest neighbours
        algorithm for a single run.

    kind_sel : str, optional (default='all')
        Strategy to use in order to exclude samples.

        - If ``'all'``, all neighbours will have to agree with the samples of
          interest to not be excluded.
        - If ``'mode'``, the majority vote of the neighbours will be used in
          order to exclude a sample.

    n_jobs : int, optional (default=1)
        The number of thread to open when it is possible.

    Notes
    -----
    The method is based on [1]_. A one-vs.-rest scheme is used when
    sampling a class as proposed in [1]_.

    Supports mutli-class resampling.

    See
    :ref:`sphx_glr_auto_examples_pipeline_plot_pipeline_classification.py` and
    :ref:`sphx_glr_auto_examples_under-sampling_plot_enn_renn_allknn.py`.

    See also
    --------
    CondensedNearestNeighbour, EditedNearestNeighbours, AllKNN

    References
    ----------
    .. [1] I. Tomek, "An Experiment with the Edited Nearest-Neighbor
       Rule," IEEE Transactions on Systems, Man, and Cybernetics, vol. 6(6),
       pp. 448-452, June 1976.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import \
RepeatedEditedNearestNeighbours # doctest : +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> renn = RepeatedEditedNearestNeighbours(random_state=42)
    >>> X_res, y_res = renn.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({1: 887, 0: 100})

    """

    def __init__(self,
                 ratio='auto',
                 return_indices=False,
                 random_state=None,
                 size_ngh=None,
                 n_neighbors=3,
                 max_iter=100,
                 kind_sel='all',
                 n_jobs=1):
        super(RepeatedEditedNearestNeighbours, self).__init__(
            ratio=ratio, random_state=random_state)
        self.return_indices = return_indices
        self.size_ngh = size_ngh
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.n_jobs = n_jobs
        self.max_iter = max_iter

    def _validate_estimator(self):
        """Private function to create the NN estimator"""
        if self.max_iter < 2:
            raise ValueError('max_iter must be greater than 1.'
                             ' Got {} instead.'.format(type(self.max_iter)))

        self.nn_ = check_neighbors_object('n_neighbors', self.n_neighbors,
                                          additional_neighbor=1)

        self.enn_ = EditedNearestNeighbours(ratio=self.ratio,
                                            return_indices=self.return_indices,
                                            random_state=self.random_state,
                                            n_neighbors=self.nn_,
                                            kind_sel=self.kind_sel,
                                            n_jobs=self.n_jobs)

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

        X_, y_ = X, y
        if self.return_indices:
            idx_under = np.arange(X.shape[0], dtype=int)
        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)

        for n_iter in range(self.max_iter):

            prev_len = y_.shape[0]
            if self.return_indices:
                X_enn, y_enn, idx_enn = self.enn_.fit_sample(X_, y_)
            else:
                X_enn, y_enn = self.enn_.fit_sample(X_, y_)

            # Check the stopping criterion
            # 1. If there is no changes for the vector y
            # 2. If the number of samples in the other class become inferior to
            # the number of samples in the majority class
            # 3. If one of the class is disappearing

            # Case 1
            b_conv = (prev_len == y_enn.shape[0])

            # Case 2
            stats_enn = Counter(y_enn)
            count_non_min = np.array([
                val for val, key in zip(stats_enn.values(), stats_enn.keys())
                if key != class_minority
            ])
            b_min_bec_maj = np.any(count_non_min <
                                   target_stats[class_minority])

            # Case 3
            b_remove_maj_class = (len(stats_enn) < len(target_stats))

            X_, y_, = X_enn, y_enn
            if self.return_indices:
                idx_under = idx_under[idx_enn]

            if b_conv or b_min_bec_maj or b_remove_maj_class:
                if b_conv:
                    if self.return_indices:
                        X_, y_, = X_enn, y_enn
                        idx_under = idx_under[idx_enn]
                    else:
                        X_, y_, = X_enn, y_enn
                break

        X_resampled, y_resampled = X_, y_

        if self.return_indices:
            return X_resampled, y_resampled, idx_under
        else:
            return X_resampled, y_resampled


class AllKNN(BaseCleaningSampler):
    """Class to perform under-sampling based on the AllKNN method.

    Read more in the :ref:`User Guide <edited_nearest_neighbors>`.

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

    kind_sel : str, optional (default='all')
        Strategy to use in order to exclude samples.

        - If ``'all'``, all neighbours will have to agree with the samples of
          interest to not be excluded.
        - If ``'mode'``, the majority vote of the neighbours will be used in
          order to exclude a sample.

    allow_minority : bool, optional (default=False)
        If ``True``, it allows the majority classes to become the minority
        class without early stopping.

        .. versionadded:: 0.3

    n_jobs : int, optional (default=1)
        The number of thread to open when it is possible.

    Notes
    -----
    The method is based on [1]_.

    Supports mutli-class resampling. A one-vs.-rest scheme is used when
    sampling a class as proposed in [1]_.

    See :ref:`sphx_glr_auto_examples_under-sampling_plot_enn_renn_allknn.py`.

    See also
    --------
    CondensedNearestNeighbour, EditedNearestNeighbours,
    RepeatedEditedNearestNeighbours

    References
    ----------
    .. [1] I. Tomek, "An Experiment with the Edited Nearest-Neighbor
       Rule," IEEE Transactions on Systems, Man, and Cybernetics, vol. 6(6),
       pp. 448-452, June 1976.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import \
AllKNN # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> allknn = AllKNN(random_state=42)
    >>> X_res, y_res = allknn.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({1: 887, 0: 100})

    """

    def __init__(self,
                 ratio='auto',
                 return_indices=False,
                 random_state=None,
                 size_ngh=None,
                 n_neighbors=3,
                 kind_sel='all',
                 allow_minority=False,
                 n_jobs=1):
        super(AllKNN, self).__init__(ratio=ratio, random_state=random_state)
        self.return_indices = return_indices
        self.size_ngh = size_ngh
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.allow_minority = allow_minority
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Create objects required by AllKNN"""
        if self.kind_sel not in SEL_KIND:
            raise NotImplementedError

        self.nn_ = check_neighbors_object('n_neighbors', self.n_neighbors,
                                          additional_neighbor=1)

        self.enn_ = EditedNearestNeighbours(ratio=self.ratio,
                                            return_indices=self.return_indices,
                                            random_state=self.random_state,
                                            n_neighbors=self.nn_,
                                            kind_sel=self.kind_sel,
                                            n_jobs=self.n_jobs)

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

        X_, y_ = X, y
        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)

        if self.return_indices:
            idx_under = np.arange(X.shape[0], dtype=int)

        for curr_size_ngh in range(1, self.nn_.n_neighbors):
            self.enn_.n_neighbors = curr_size_ngh

            if self.return_indices:
                X_enn, y_enn, idx_enn = self.enn_.fit_sample(X_, y_)
            else:
                X_enn, y_enn = self.enn_.fit_sample(X_, y_)

            # Check the stopping criterion
            # 1. If the number of samples in the other class become inferior to
            # the number of samples in the majority class
            # 2. If one of the class is disappearing
            # Case 1
            stats_enn = Counter(y_enn)
            count_non_min = np.array([
                val for val, key in zip(stats_enn.values(), stats_enn.keys())
                if key != class_minority
            ])
            b_min_bec_maj = np.any(count_non_min <
                                   target_stats[class_minority])
            if self.allow_minority:
                # overwrite b_min_bec_maj
                b_min_bec_maj = False

            # Case 2
            b_remove_maj_class = (len(stats_enn) < len(target_stats))

            X_, y_, = X_enn, y_enn
            if self.return_indices:
                idx_under = idx_under[idx_enn]

            if b_min_bec_maj or b_remove_maj_class:
                break

        X_resampled, y_resampled = X_, y_

        if self.return_indices:
            return X_resampled, y_resampled, idx_under
        else:
            return X_resampled, y_resampled
