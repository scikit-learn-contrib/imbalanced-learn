"""Class to perform over-sampling using SMOTE."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Fernando Nogueira
#          Christos Aridas
# License: MIT

from __future__ import division

import numpy as np

from scipy import sparse

from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.utils import check_random_state, safe_indexing
from sklearn.metrics.pairwise import pairwise_distances

from .base import BaseOverSampler
from ..exceptions import raise_isinstance_error
from ..utils import check_neighbors_object
from ..utils import Substitution
from ..utils._docstring import _random_state_docstring

SMOTE_KIND = ('regular', 'borderline1', 'borderline2', 'svm', 'kmeans')


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring)
class SMOTE(BaseOverSampler):
    """Class to perform over-sampling using SMOTE.

    This object is an implementation of SMOTE - Synthetic Minority
    Over-sampling Technique, and the variants Borderline SMOTE 1, 2 and
    SVM-SMOTE.

    Read more in the :ref:`User Guide <smote_adasyn>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    k_neighbors : int or object, optional (default=5)
        If ``int``, number of nearest neighbours to used to construct synthetic
        samples.  If object, an estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.

    m_neighbors : int or object, optional (default=10)
        If int, number of nearest neighbours to use to determine if a minority
        sample is in danger. Used with ``kind={{'borderline1', 'borderline2',
        'svm'}}``.  If object, an estimator that inherits
        from :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used
        to find the k_neighbors.

    out_step : float, optional (default=0.5)
        Step size when extrapolating. Used with ``kind='svm'``.

    kind : str, optional (default='regular')
        The type of SMOTE algorithm to use one of the following options:
        ``'regular'``, ``'borderline1'``, ``'borderline2'``, ``'svm'``, ``'kmeans'``.

    svm_estimator : object, optional (default=SVC())
        If ``kind='svm'``, a parametrized :class:`sklearn.svm.SVC`
        classifier can be passed.

    n_kmeans_clusters: int, optional (default=10)
        If ``kind='kmeans'``, the number of clusters that is the be used by the
        k-means algorithm for sample identification.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    ratio : str, dict, or callable
        .. deprecated:: 0.4
           Use the parameter ``sampling_strategy`` instead. It will be removed
           in 0.6.

    Notes
    -----
    See the original papers: [1]_, [2]_, [3]_ for more details.

    Supports multi-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    See
    :ref:`sphx_glr_auto_examples_applications_plot_over_sampling_benchmark_lfw.py`,
    :ref:`sphx_glr_auto_examples_evaluation_plot_classification_report.py`,
    :ref:`sphx_glr_auto_examples_evaluation_plot_metrics.py`,
    :ref:`sphx_glr_auto_examples_model_selection_plot_validation_curve.py`,
    :ref:`sphx_glr_auto_examples_over-sampling_plot_comparison_over_sampling.py`,
    and :ref:`sphx_glr_auto_examples_over-sampling_plot_smote.py`.

    See also
    --------
    ADASYN : Over-sample using ADASYN.

    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, "SMOTE:
       synthetic minority over-sampling technique," Journal of artificial
       intelligence research, 321-357, 2002.

    .. [2] H. Han, W. Wen-Yuan, M. Bing-Huan, "Borderline-SMOTE: a new
       over-sampling method in imbalanced data sets learning," Advances in
       intelligent computing, 878-887, 2005.

    .. [3] H. M. Nguyen, E. W. Cooper, K. Kamei, "Borderline over-sampling for
       imbalanced data classification," International Journal of Knowledge
       Engineering and Soft Data Paradigms, 3(1), pp.4-21, 2001.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.over_sampling import \
SMOTE # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> sm = SMOTE(random_state=42)
    >>> X_res, y_res = sm.fit_sample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 900, 1: 900}})

    """

    def __init__(self,
                 sampling_strategy='auto',
                 random_state=None,
                 k_neighbors=5,
                 m_neighbors=10,
                 out_step=0.5,
                 kind='regular',
                 svm_estimator=None,
                 n_kmeans_clusters=10,
                 n_jobs=1,
                 ratio=None):
        super(SMOTE, self).__init__(
            sampling_strategy=sampling_strategy, ratio=ratio)
        self.random_state = random_state
        self.kind = kind
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors
        self.out_step = out_step
        self.svm_estimator = svm_estimator
        self.n_kmeans_clusters = n_kmeans_clusters
        self.n_jobs = n_jobs

    def _in_danger_noise(self, samples, target_class, y, kind='danger'):
        """Estimate if a set of sample are in danger or noise.

        Parameters
        ----------
        samples : {array-like, sparse matrix}, shape (n_samples, n_features)
            The samples to check if either they are in danger or not.

        target_class : int or str,
            The target corresponding class being over-sampled.

        y : array-like, shape (n_samples,)
            The true label in order to check the neighbour labels.

        kind : str, optional (default='danger')
            The type of classification to use. Can be either:

            - If 'danger', check if samples are in danger,
            - If 'noise', check if samples are noise.

        Returns
        -------
        output : ndarray, shape (n_samples,)
            A boolean array where True refer to samples in danger or noise.

        """
        x = self.nn_m_.kneighbors(samples, return_distance=False)[:, 1:]
        nn_label = (y[x] != target_class).astype(int)
        n_maj = np.sum(nn_label, axis=1)

        if kind == 'danger':
            # Samples are in danger for m/2 <= m' < m
            return np.bitwise_and(n_maj >= (self.nn_m_.n_neighbors - 1) / 2,
                                  n_maj < self.nn_m_.n_neighbors - 1)
        elif kind == 'noise':
            # Samples are noise for m = m'
            return n_maj == self.nn_m_.n_neighbors - 1
        else:
            raise NotImplementedError

    def _make_samples(self,
                      X,
                      y_type,
                      nn_data,
                      nn_num,
                      n_samples,
                      step_size=1.):
        """A support function that returns artificial samples constructed along
        the line connecting nearest neighbours.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Points from which the points will be created.

        y_type : str or int
            The minority target value, just so the function can return the
            target values for the synthetic variables with correct length in
            a clear format.

        nn_data : ndarray, shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used

        nn_num : ndarray, shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in nn_data.

        n_samples : int
            The number of samples to generate.

        step_size : float, optional (default=1.)
            The step size to create samples.

        Returns
        -------
        X_new : {ndarray, sparse matrix}, shape (n_samples_new, n_features)
            Synthetically generated samples.

        y_new : ndarray, shape (n_samples_new,)
            Target values for synthetic samples.

        """
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(
            low=0, high=len(nn_num.flatten()), size=n_samples)
        steps = step_size * random_state.uniform(size=n_samples)
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])

        if sparse.issparse(X):
            row_indices, col_indices, samples = [], [], []
            for i, (row, col, step) in enumerate(zip(rows, cols, steps)):
                if X[row].nnz:
                    sample = X[row] - step * (
                        X[row] - nn_data[nn_num[row, col]])
                    row_indices += [i] * len(sample.indices)
                    col_indices += sample.indices.tolist()
                    samples += sample.data.tolist()
        else:
            X_new = np.zeros((n_samples, X.shape[1]))
            for i, (row, col, step) in enumerate(zip(rows, cols, steps)):
                X_new[i] = X[row] - step * (X[row] - nn_data[nn_num[row, col]])

        y_new = np.array([y_type] * len(samples_indices))

        if sparse.issparse(X):
            return (sparse.csr_matrix((samples, (row_indices, col_indices)),
                                      [len(samples_indices), X.shape[1]]),
                    y_new)
        else:
            return X_new, y_new

    def _validate_estimator(self):
        """Create the necessary objects for SMOTE."""

        if self.kind not in SMOTE_KIND:
            raise ValueError('Unknown kind for SMOTE algorithm.'
                             ' Choices are {}. Got {} instead.'.format(
                                 SMOTE_KIND, self.kind))

        self.nn_k_ = check_neighbors_object(
            'k_neighbors', self.k_neighbors, additional_neighbor=1)
        self.nn_k_.set_params(**{'n_jobs': self.n_jobs})

        if self.kind != 'regular':
            self.nn_m_ = check_neighbors_object(
                'm_neighbors', self.m_neighbors, additional_neighbor=1)
            self.nn_m_.set_params(**{'n_jobs': self.n_jobs})

        if self.kind == 'svm':
            if self.svm_estimator is None:
                self.svm_estimator_ = SVC(random_state=self.random_state)
            elif isinstance(self.svm_estimator, SVC):
                self.svm_estimator_ = self.svm_estimator
            else:
                raise_isinstance_error('svm_estimator', [SVC],
                                       self.svm_estimator)

    def _sample_regular(self, X, y):
        """Resample the dataset using the regular SMOTE implementation.

        Use the regular SMOTE algorithm proposed in [1]_.

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

        References
        ----------
        .. [1] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, "SMOTE:
           synthetic minority over-sampling technique," Journal of artificial
           intelligence research, 321-357, 2002.

        """

        X_resampled = X.copy()
        y_resampled = y.copy()

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = safe_indexing(X, target_class_indices)

            self.nn_k_.fit(X_class)
            nns = self.nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]
            X_new, y_new = self._make_samples(X_class, class_sample, X_class,
                                              nns, n_samples, 1.0)

            if sparse.issparse(X_new):
                X_resampled = sparse.vstack([X_resampled, X_new])
            else:
                X_resampled = np.vstack((X_resampled, X_new))
            y_resampled = np.hstack((y_resampled, y_new))

        return X_resampled, y_resampled

    def _sample_borderline(self, X, y):
        """Resample the dataset using the borderline SMOTE implementation.

        Use the borderline SMOTE algorithm proposed in [2]_. Two methods can be
        used: (i) borderline-1 or (ii) borderline-2. A nearest-neighbours
        algorithm is used to determine the samples forming the boundaries and
        will create samples next to those features depending on some criterion.

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

        References
        ----------
        .. [2] H. Han, W. Wen-Yuan, M. Bing-Huan, "Borderline-SMOTE: a new
           over-sampling method in imbalanced data sets learning," Advances in
           intelligent computing, 878-887, 2005.

        """
        X_resampled = X.copy()
        y_resampled = y.copy()

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = safe_indexing(X, target_class_indices)

            self.nn_m_.fit(X)
            danger_index = self._in_danger_noise(
                X_class, class_sample, y, kind='danger')
            if not any(danger_index):
                continue

            self.nn_k_.fit(X_class)
            nns = self.nn_k_.kneighbors(
                safe_indexing(X_class, danger_index),
                return_distance=False)[:, 1:]

            # divergence between borderline-1 and borderline-2
            if self.kind == 'borderline1':
                # Create synthetic samples for borderline points.
                X_new, y_new = self._make_samples(
                    safe_indexing(X_class, danger_index), class_sample,
                    X_class, nns, n_samples)
                if sparse.issparse(X_new):
                    X_resampled = sparse.vstack([X_resampled, X_new])
                else:
                    X_resampled = np.vstack((X_resampled, X_new))
                y_resampled = np.hstack((y_resampled, y_new))

            else:
                random_state = check_random_state(self.random_state)
                fractions = random_state.beta(10, 10)

                # only minority
                X_new_1, y_new_1 = self._make_samples(
                    safe_indexing(X_class, danger_index),
                    class_sample,
                    X_class,
                    nns,
                    int(fractions * (n_samples + 1)),
                    step_size=1.)

                # we use a one-vs-rest policy to handle the multiclass in which
                # new samples will be created considering not only the majority
                # class but all over classes.
                X_new_2, y_new_2 = self._make_samples(
                    safe_indexing(X_class, danger_index),
                    class_sample,
                    safe_indexing(X, np.flatnonzero(y != class_sample)),
                    nns,
                    int((1 - fractions) * n_samples),
                    step_size=0.5)

                if sparse.issparse(X_resampled):
                    X_resampled = sparse.vstack(
                        [X_resampled, X_new_1, X_new_2])
                else:
                    X_resampled = np.vstack((X_resampled, X_new_1, X_new_2))
                y_resampled = np.hstack((y_resampled, y_new_1, y_new_2))

        return X_resampled, y_resampled

    def _sample_svm(self, X, y):
        """Resample the dataset using the SVM SMOTE implementation.

        Use the SVM SMOTE algorithm proposed in [3]_. A SVM classifier detect
        support vectors to get a notion of the boundary.

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

        References
        ----------
        .. [3] H. M. Nguyen, E. W. Cooper, K. Kamei, "Borderline over-sampling
           for imbalanced data classification," International Journal of
           Knowledge Engineering and Soft Data Paradigms, 3(1), pp.4-21, 2001.

        """
        random_state = check_random_state(self.random_state)
        X_resampled = X.copy()
        y_resampled = y.copy()

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = safe_indexing(X, target_class_indices)

            self.svm_estimator_.fit(X, y)
            support_index = self.svm_estimator_.support_[y[
                self.svm_estimator_.support_] == class_sample]
            support_vector = safe_indexing(X, support_index)

            self.nn_m_.fit(X)
            noise_bool = self._in_danger_noise(
                support_vector, class_sample, y, kind='noise')
            support_vector = safe_indexing(
                support_vector, np.flatnonzero(np.logical_not(noise_bool)))
            danger_bool = self._in_danger_noise(
                support_vector, class_sample, y, kind='danger')
            safety_bool = np.logical_not(danger_bool)

            self.nn_k_.fit(X_class)
            fractions = random_state.beta(10, 10)
            if np.count_nonzero(danger_bool) > 0:
                nns = self.nn_k_.kneighbors(
                    safe_indexing(support_vector, np.flatnonzero(danger_bool)),
                    return_distance=False)[:, 1:]

                X_new_1, y_new_1 = self._make_samples(
                    safe_indexing(support_vector, np.flatnonzero(danger_bool)),
                    class_sample,
                    X_class,
                    nns,
                    int(fractions * (n_samples + 1)),
                    step_size=1.)

            if np.count_nonzero(safety_bool) > 0:
                nns = self.nn_k_.kneighbors(
                    safe_indexing(support_vector, np.flatnonzero(safety_bool)),
                    return_distance=False)[:, 1:]

                X_new_2, y_new_2 = self._make_samples(
                    safe_indexing(support_vector, np.flatnonzero(safety_bool)),
                    class_sample,
                    X_class,
                    nns,
                    int((1 - fractions) * n_samples),
                    step_size=-self.out_step)

            if (np.count_nonzero(danger_bool) > 0 and
                    np.count_nonzero(safety_bool) > 0):
                if sparse.issparse(X_resampled):
                    X_resampled = sparse.vstack(
                        [X_resampled, X_new_1, X_new_2])
                else:
                    X_resampled = np.vstack((X_resampled, X_new_1, X_new_2))
                y_resampled = np.concatenate(
                    (y_resampled, y_new_1, y_new_2), axis=0)
            elif np.count_nonzero(danger_bool) == 0:
                if sparse.issparse(X_resampled):
                    X_resampled = sparse.vstack([X_resampled, X_new_2])
                else:
                    X_resampled = np.vstack((X_resampled, X_new_2))
                y_resampled = np.concatenate((y_resampled, y_new_2), axis=0)
            elif np.count_nonzero(safety_bool) == 0:
                if sparse.issparse(X_resampled):
                    X_resampled = sparse.vstack([X_resampled, X_new_1])
                else:
                    X_resampled = np.vstack((X_resampled, X_new_1))
                y_resampled = np.concatenate((y_resampled, y_new_1), axis=0)

        return X_resampled, y_resampled

    def _find_cluster_sparsity(self, X):
        """ Finds the sparsity of a cluster of samples. The sparsity is
         calculated according to the method described in [4]_. """

        euclidean_distances = pairwise_distances(
            X, metric="euclidean", n_jobs=self.n_jobs
        )

        # Negate diagonal elements.
        for ind in range(X.shape[0]):
            euclidean_distances[ind, ind] = 0

        non_diag_elements = (len(X) ** 2) - len(X)
        mean_distance = euclidean_distances.sum() / non_diag_elements

        density = len(X) / (mean_distance ** 2)
        sparsity = 1 / density
        return sparsity

    def _sample_kmeans(self, X, y):
        """Resample the dataset using the SMOTE K-Means implementation.

        Use the SMOTE K-Means algorithm proposed in [4]_. K-Means clustering
        is used to select samples for over sampling.

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

        References
        ----------
        .. [4] H. M. Nguyen, E. W. Cooper, K. Kamei, "Borderline over-sampling
           for imbalanced data classification," International Journal of
           Knowledge Engineering and Soft Data Paradigms, 3(1), pp.4-21, 2001.

        """
        random_state = check_random_state(self.random_state)
        X_resampled = X.copy()
        y_resampled = y.copy()

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = safe_indexing(X, target_class_indices)

            km = KMeans(
                self.n_kmeans_clusters,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )
            X_clusters = km.fit_predict(X)

            valid_clusters = []
            cluster_sparsities = []

            # Identify clusters where class_sample is the majority
            for cluster_n in range(self.n_kmeans_clusters):
                cluster_index = np.flatnonzero(X_clusters == cluster_n)

                X_cluster = safe_indexing(X, cluster_index)
                y_cluster = safe_indexing(y, cluster_index)

                cluster_class_mean = (y_cluster == class_sample).mean()

                X_cluster_class = safe_indexing(
                    X_cluster,
                    np.flatnonzero(y_cluster == class_sample)
                )

                if len(X_cluster_class) < self.k_neighbors + 1:
                    continue

                if cluster_class_mean < 0.5:
                    continue

                valid_clusters.append(cluster_index)
                cluster_sparsities.append(
                    self._find_cluster_sparsity(X_cluster_class)
                )

            cluster_weights = [
                cs / sum(cluster_sparsities) for cs in cluster_sparsities
            ]

            for cluster_n in range(len(valid_clusters)):
                X_cluster = safe_indexing(X, valid_clusters[cluster_n])
                y_cluster = safe_indexing(y, valid_clusters[cluster_n])

                X_cluster_class = safe_indexing(
                    X_cluster, np.flatnonzero(y_cluster == class_sample)
                )

                self.nn_k_.fit(X_cluster_class)

                nns = self.nn_k_.kneighbors(
                    X_cluster_class, return_distance=False
                )[:, 1:]

                c_n_samples = int(n_samples * cluster_weights[cluster_n])
                X_new, y_new = self._make_samples(
                    X_cluster_class,
                    class_sample,
                    X_class,
                    nns,
                    c_n_samples,
                    1.0
                )

                if sparse.issparse(X_new):
                    X_resampled = sparse.vstack([X_resampled, X_new])
                else:
                    X_resampled = np.vstack((X_resampled, X_new))
                y_resampled = np.hstack((y_resampled, y_new))

        return X_resampled, y_resampled

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

        if self.kind == 'regular':
            return self._sample_regular(X, y)
        elif self.kind == 'borderline1' or self.kind == 'borderline2':
            return self._sample_borderline(X, y)
        elif self.kind == 'svm':
            return self._sample_svm(X, y)
        elif self.kind == 'kmeans':
            return self._sample_kmeans(X, y)
