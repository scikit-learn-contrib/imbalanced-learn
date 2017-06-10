"""Class to perform under-sampling based on one-sided selection method."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import division

from collections import Counter

import numpy as np
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.utils import check_random_state

from ..base import BaseCleaningSampler
from .tomek_links import TomekLinks
from ...utils.deprecation import deprecate_parameter


class OneSidedSelection(BaseCleaningSampler):
    """Class to perform under-sampling based on one-sided selection method.

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

    size_ngh : int, optional (default=None)
        Size of the neighbourhood to consider to compute the average
        distance to the minority point samples.

        .. deprecated:: 0.2
           ``size_ngh`` is deprecated from 0.2 and will be replaced in 0.4
           Use ``n_neighbors`` instead.

    n_neighbors : int or object, optional (default=\
KNeighborsClassifier(n_neighbors=1))
        If ``int``, size of the neighbourhood to consider to compute the
        average distance to the minority point samples.  If object, an object
        inherited from :class:`sklearn.neigbors.KNeighborsClassifier` should be
        passed.

    n_seeds_S : int, optional (default=1)
        Number of samples to extract in order to build the set S.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    Notes
    -----
    The method is based on [1]_.

    Supports mutli-class resampling.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import \
    OneSidedSelection # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> oss = OneSidedSelection(random_state=42)
    >>> X_res, y_res = oss.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({1: 495, 0: 100})

    References
    ----------
    .. [1] M. Kubat, S. Matwin, "Addressing the curse of imbalanced training
       sets: one-sided selection," In ICML, vol. 97, pp. 179-186, 1997.

    """

    def __init__(self,
                 ratio='auto',
                 return_indices=False,
                 random_state=None,
                 size_ngh=None,
                 n_neighbors=None,
                 n_seeds_S=1,
                 n_jobs=1):
        super(OneSidedSelection, self).__init__(ratio=ratio,
                                                random_state=random_state)
        self.return_indices = return_indices
        self.size_ngh = size_ngh
        self.n_neighbors = n_neighbors
        self.n_seeds_S = n_seeds_S
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Private function to create the NN estimator"""
        # FIXME: Deprecated in 0.2. To be removed in 0.4.
        deprecate_parameter(self, '0.2', 'size_ngh', 'n_neighbors')
        if self.n_neighbors is None:
            self.estimator_ = KNeighborsClassifier(
                n_neighbors=1, n_jobs=self.n_jobs)
        elif isinstance(self.n_neighbors, int):
            self.estimator_ = KNeighborsClassifier(
                n_neighbors=self.n_neighbors, n_jobs=self.n_jobs)
        elif isinstance(self.n_neighbors, KNeighborsClassifier):
            self.estimator_ = self.n_neighbors
        else:
            raise ValueError('`n_neighbors` has to be a int or an object'
                             ' inhereited from KNeighborsClassifier.'
                             ' Got {} instead.'.format(type(self.n_neighbors)))

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

        random_state = check_random_state(self.random_state)
        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)

        X_resampled = np.empty((0, X.shape[1]), dtype=X.dtype)
        y_resampled = np.empty((0, ), dtype=y.dtype)
        if self.return_indices:
            idx_under = np.empty((0, ), dtype=int)

        for target_class in np.unique(y):
            if target_class in self.ratio_.keys():
                # select a sample from the current class
                idx_maj = np.flatnonzero(y == target_class)
                idx_maj_sample = idx_maj[random_state.randint(
                        low=0, high=target_stats[target_class],
                        size=self.n_seeds_S)]
                maj_sample = X[idx_maj_sample]

                # create the set composed of all minority samples and one
                # sample from the current class.
                C_x = np.append(X[y == class_minority], maj_sample, axis=0)
                C_y = np.append(y[y == class_minority], [target_class] *
                                self.n_seeds_S)

                # create the set S with removing the seed from S
                # since that it will be added anyway
                idx_maj_extracted = np.delete(idx_maj, idx_maj_sample, axis=0)
                S_x = X[idx_maj_extracted]
                S_y = y[idx_maj_extracted]
                self.estimator_.fit(C_x, C_y)
                pred_S_y = self.estimator_.predict(S_x)

                sel_x = S_x[np.flatnonzero(pred_S_y != S_y), :]
                sel_y = S_y[np.flatnonzero(pred_S_y != S_y)]
                if self.return_indices:
                    idx_tmp = idx_maj_extracted[
                        np.flatnonzero(pred_S_y != S_y)]
                    idx_under = np.concatenate(
                        (idx_under, idx_maj_sample, idx_tmp), axis=0)
                X_resampled = np.concatenate(
                    (X_resampled, maj_sample, sel_x), axis=0)
                y_resampled = np.concatenate(
                    (y_resampled, [target_class] * self.n_seeds_S, sel_y),
                    axis=0)
            else:
                X_resampled = np.concatenate(
                    (X_resampled, X[y == target_class]), axis=0)
                y_resampled = np.concatenate(
                    (y_resampled, y[y == target_class]), axis=0)
                if self.return_indices:
                    idx_under = np.concatenate(
                        (idx_under, np.flatnonzero(y == target_class)), axis=0)

        # find the nearest neighbour of every point
        nn = NearestNeighbors(n_neighbors=2, n_jobs=self.n_jobs)
        nn.fit(X_resampled)
        nns = nn.kneighbors(X_resampled, return_distance=False)[:, 1]

        links = TomekLinks.is_tomek(y_resampled, nns,
                                    [c for c in np.unique(y)
                                     if (c != class_minority and
                                         c in self.ratio_.keys())])
        if self.return_indices:
            return (X_resampled[np.logical_not(links)],
                    y_resampled[np.logical_not(links)],
                    idx_under[np.logical_not(links)])
        else:
            return (X_resampled[np.logical_not(links)],
                    y_resampled[np.logical_not(links)])
