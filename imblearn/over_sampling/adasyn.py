"""Class to perform random over-sampling."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from __future__ import division

import numpy as np
from scipy import sparse

from sklearn.utils import check_random_state, safe_indexing

from .base import BaseOverSampler
from ..utils import check_neighbors_object
from ..utils.deprecation import deprecate_parameter


class ADASYN(BaseOverSampler):
    """Perform over-sampling using ADASYN.

    Perform over-sampling using Adaptive Synthetic Sampling Approach for
    Imbalanced Learning.

    Read more in the :ref:`User Guide <smote_adasyn>`.

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

    random_state : int, RandomState instance or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number
        generator; If ``RandomState`` instance, random_state is the random
        number generator; If ``None``, the random number generator is the
        ``RandomState`` instance used by ``np.random``.

    k : int, optional (default=None)
        Number of nearest neighbours to used to construct synthetic samples.

        .. deprecated:: 0.2
           ``k`` is deprecated from 0.2 and will be replaced in 0.4
           Use ``n_neighbors`` instead.

    n_neighbors : int int or object, optional (default=5)
        If ``int``, number of nearest neighbours to used to construct synthetic
        samples.  If object, an estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.

    n_jobs : int, optional (default=1)
        Number of threads to run the algorithm when it is possible.

    Notes
    -----
    The implementation is based on [1]_.

    Supports mutli-class resampling. A one-vs.-rest scheme is used.

    See
    :ref:`sphx_glr_auto_examples_applications_plot_over_sampling_benchmark_lfw.py`,
    :ref:`sphx_glr_auto_examples_over-sampling_plot_adasyn.py` and
    :ref:`sphx_glr_auto_examples_over-sampling_plot_comparison_over_sampling.py`.

    See also
    --------
    SMOTE : Over-sample using SMOTE.

    References
    ----------
    .. [1] He, Haibo, Yang Bai, Edwardo A. Garcia, and Shutao Li. "ADASYN:
       Adaptive synthetic sampling approach for imbalanced learning," In IEEE
       International Joint Conference on Neural Networks (IEEE World Congress
       on Computational Intelligence), pp. 1322-1328, 2008.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.over_sampling import \
ADASYN # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000,
    ... random_state=10)
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> ada = ADASYN(random_state=42)
    >>> X_res, y_res = ada.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res)))
    Resampled dataset shape Counter({0: 904, 1: 900})

    """

    def __init__(self,
                 ratio='auto',
                 random_state=None,
                 k=None,
                 n_neighbors=5,
                 n_jobs=1):
        super(ADASYN, self).__init__(ratio=ratio, random_state=random_state)
        self.k = k
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Create the necessary objects for ADASYN"""
        # FIXME: Deprecated in 0.2. To be removed in 0.4.
        deprecate_parameter(self, '0.2', 'k', 'n_neighbors')

        self.nn_ = check_neighbors_object('n_neighbors', self.n_neighbors,
                                          additional_neighbor=1)
        self.nn_.set_params(**{'n_jobs': self.n_jobs})

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


        """
        self._validate_estimator()
        random_state = check_random_state(self.random_state)

        X_resampled = X.copy()
        y_resampled = y.copy()

        for class_sample, n_samples in self.ratio_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = safe_indexing(X, target_class_indices)

            self.nn_.fit(X)
            _, nn_index = self.nn_.kneighbors(X_class)
            # The ratio is computed using a one-vs-rest manner. Using majority
            # in multi-class would lead to slightly different results at the
            # cost of introducing a new parameter.
            ratio_nn = (np.sum(y[nn_index[:, 1:]] != class_sample, axis=1) /
                        (self.nn_.n_neighbors - 1))
            if not np.sum(ratio_nn):
                raise RuntimeError('Not any neigbours belong to the majority'
                                   ' class. This case will induce a NaN case'
                                   ' with a division by zero. ADASYN is not'
                                   ' suited for this specific dataset.'
                                   ' Use SMOTE instead.')
            ratio_nn /= np.sum(ratio_nn)
            n_samples_generate = np.rint(ratio_nn * n_samples).astype(int)
            if not np.sum(n_samples_generate):
                raise ValueError("No samples will be generated with the"
                                 " provided ratio settings.")

            # the nearest neighbors need to be fitted only on the current class
            # to find the class NN to generate new samples
            self.nn_.fit(X_class)
            _, nn_index = self.nn_.kneighbors(X_class)

            if sparse.issparse(X):
                row_indices, col_indices, samples = [], [], []
                n_samples_generated = 0
                for x_i, x_i_nn, num_sample_i in zip(X_class, nn_index,
                                                     n_samples_generate):
                    if num_sample_i == 0:
                        continue
                    nn_zs = random_state.randint(
                        1, high=self.nn_.n_neighbors, size=num_sample_i)
                    steps = random_state.uniform(size=len(nn_zs))
                    if x_i.nnz:
                        for step, nn_z in zip(steps, nn_zs):
                            sample = (x_i +
                                      step * (X_class[x_i_nn[nn_z], :] - x_i))
                            row_indices += ([n_samples_generated] *
                                            len(sample.indices))
                            col_indices += sample.indices.tolist()
                            samples += sample.data.tolist()
                            n_samples_generated += 1
                X_new = (sparse.csr_matrix((samples,
                                            (row_indices, col_indices)),
                                           [np.sum(n_samples_generate),
                                            X.shape[1]]))
                y_new = np.array([class_sample] * np.sum(n_samples_generate))
            else:
                x_class_gen = []
                for x_i, x_i_nn, num_sample_i in zip(X_class, nn_index,
                                                     n_samples_generate):
                    if num_sample_i == 0:
                        continue
                    nn_zs = random_state.randint(
                        1, high=self.nn_.n_neighbors, size=num_sample_i)
                    steps = random_state.uniform(size=len(nn_zs))
                    x_class_gen.append([x_i +
                                        step * (X_class[x_i_nn[nn_z], :] - x_i)
                                        for step, nn_z in zip(steps, nn_zs)])

                X_new = np.concatenate(x_class_gen)
                y_new = np.array([class_sample] * np.sum(n_samples_generate))

            if sparse.issparse(X_new):
                X_resampled = sparse.vstack([X_resampled, X_new])
            else:
                X_resampled = np.vstack((X_resampled, X_new))
            y_resampled = np.hstack((y_resampled, y_new))

        return X_resampled, y_resampled
