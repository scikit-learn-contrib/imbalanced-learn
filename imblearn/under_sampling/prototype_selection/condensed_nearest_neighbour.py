"""Class to perform under-sampling based on the condensed nearest neighbour
method."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import division

from collections import Counter

import numpy as np

from scipy.sparse import issparse

from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_random_state, safe_indexing

from ..base import BaseCleaningSampler
from ...utils.deprecation import deprecate_parameter


class CondensedNearestNeighbour(BaseCleaningSampler):
    """Class to perform under-sampling based on the condensed nearest neighbour
    method.

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

    n_neighbors : int or object, optional (default=\
KNeighborsClassifier(n_neighbors=1))
        If ``int``, size of the neighbourhood to consider to compute the
        nearest neighbors. If object, an estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the nearest-neighbors.

    n_seeds_S : int, optional (default=1)
        Number of samples to extract in order to build the set S.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    Notes
    -----
    The method is based on [1]_.

    Supports mutli-class resampling. A one-vs.-rest scheme is used when
    sampling a class as proposed in [1]_.

    See
    :ref:`sphx_glr_auto_examples_under-sampling_plot_condensed_nearest_neighbour.py`.

    See also
    --------
    EditedNearestNeighbours, RepeatedEditedNearestNeighbours, AllKNN

    References
    ----------
    .. [1] P. Hart, "The condensed nearest neighbor rule,"
       In Information Theory, IEEE Transactions on, vol. 14(3),
       pp. 515-516, 1968.

    Examples
    --------

    >>> from collections import Counter #doctest: +SKIP
    >>> from sklearn.datasets import fetch_mldata #doctest: +SKIP
    >>> from imblearn.under_sampling import \
CondensedNearestNeighbour #doctest: +SKIP
    >>> pima = fetch_mldata('diabetes_scale') #doctest: +SKIP
    >>> X, y = pima['data'], pima['target'] #doctest: +SKIP
    >>> print('Original dataset shape {}'.format(Counter(y))) #doctest: +SKIP
    Original dataset shape Counter({1: 500, -1: 268}) #doctest: +SKIP
    >>> cnn = CondensedNearestNeighbour(random_state=42) #doctest: +SKIP
    >>> X_res, y_res = cnn.fit_sample(X, y) #doctest: +SKIP
    >>> print('Resampled dataset shape {}'.format(
    ... Counter(y_res))) #doctest: +SKIP
    Resampled dataset shape Counter({-1: 268, 1: 227}) #doctest: +SKIP

    """

    def __init__(self,
                 ratio='auto',
                 return_indices=False,
                 random_state=None,
                 size_ngh=None,
                 n_neighbors=None,
                 n_seeds_S=1,
                 n_jobs=1):
        super(CondensedNearestNeighbour, self).__init__(
            ratio=ratio, random_state=random_state)
        self.return_indices = return_indices
        self.size_ngh = size_ngh
        self.n_neighbors = n_neighbors
        self.n_seeds_S = n_seeds_S
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Private function to create the NN estimator"""
        # FIXME: Deprecated in 0.2. To be removed in 0.4
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

        random_state = check_random_state(self.random_state)
        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)
        idx_under = np.empty((0, ), dtype=int)

        for target_class in np.unique(y):
            if target_class in self.ratio_.keys():
                # Randomly get one sample from the majority class
                # Generate the index to select
                idx_maj = np.flatnonzero(y == target_class)
                idx_maj_sample = idx_maj[random_state.randint(
                        low=0, high=target_stats[target_class],
                        size=self.n_seeds_S)]

                # Create the set C - One majority samples and all minority
                C_indices = np.append(np.flatnonzero(y == class_minority),
                                      idx_maj_sample)
                C_x = safe_indexing(X, C_indices)
                C_y = safe_indexing(y, C_indices)

                # Create the set S - all majority samples
                S_indices = np.flatnonzero(y == target_class)
                S_x = safe_indexing(X, S_indices)
                S_y = safe_indexing(y, S_indices)

                # fit knn on C
                self.estimator_.fit(C_x, C_y)

                good_classif_label = idx_maj_sample.copy()
                # Check each sample in S if we keep it or drop it
                for idx_sam, (x_sam, y_sam) in enumerate(zip(S_x, S_y)):

                    # Do not select sample which are already well classified
                    if idx_sam in good_classif_label:
                        continue

                    # Classify on S
                    if not issparse(x_sam):
                        x_sam = x_sam.reshape(1, -1)
                    pred_y = self.estimator_.predict(x_sam)

                    # If the prediction do not agree with the true label
                    # append it in C_x
                    if y_sam != pred_y:
                        # Keep the index for later
                        idx_maj_sample = np.append(idx_maj_sample,
                                                   idx_maj[idx_sam])

                        # Update C
                        C_indices = np.append(C_indices, idx_maj[idx_sam])
                        C_x = safe_indexing(X, C_indices)
                        C_y = safe_indexing(y, C_indices)

                        # fit a knn on C
                        self.estimator_.fit(C_x, C_y)

                        # This experimental to speed up the search
                        # Classify all the element in S and avoid to test the
                        # well classified elements
                        pred_S_y = self.estimator_.predict(S_x)
                        good_classif_label = np.unique(
                            np.append(idx_maj_sample,
                                      np.flatnonzero(pred_S_y == S_y)))

                idx_under = np.concatenate((idx_under, idx_maj_sample),
                                           axis=0)
            else:
                idx_under = np.concatenate(
                    (idx_under, np.flatnonzero(y == target_class)), axis=0)

        if self.return_indices:
            return (safe_indexing(X, idx_under), safe_indexing(y, idx_under),
                    idx_under)
        else:
            return safe_indexing(X, idx_under), safe_indexing(y, idx_under)
