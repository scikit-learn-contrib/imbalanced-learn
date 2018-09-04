"""Class to perform over-sampling using SMOTE."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Fernando Nogueira
#          Christos Aridas
#          Dzianis Dudnik
# License: MIT

from __future__ import division

import types
import warnings

import numpy as np

from scipy import sparse

from sklearn.base import clone
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils import check_random_state, safe_indexing, check_array

from .base import BaseOverSampler
from ..exceptions import raise_isinstance_error
from ..utils import check_neighbors_object
from ..utils import Substitution
from ..utils._docstring import _random_state_docstring

# FIXME: remove in 0.6
SMOTE_KIND = ('regular', 'borderline1', 'borderline2', 'svm')


class BaseSMOTE(BaseOverSampler):
    """Base class for the different SMOTE algorithms."""
    def __init__(self,
                 sampling_strategy='auto',
                 random_state=None,
                 k_neighbors=5,
                 n_jobs=1,
                 ratio=None):
        super(BaseSMOTE, self).__init__(
            sampling_strategy=sampling_strategy, ratio=ratio)
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Check the NN estimators shared across the different SMOTE
        algorithms.
        """
        self.nn_k_ = check_neighbors_object(
            'k_neighbors', self.k_neighbors, additional_neighbor=1)
        self.nn_k_.set_params(**{'n_jobs': self.n_jobs})

    def _make_samples(self,
                      X,
                      y_dtype,
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

        y_dtype : dtype
            The data type of the targets.

        y_type : str or int
            The minority target value, just so the function can return the
            target values for the synthetic variables with correct length in
            a clear format.

        nn_data : ndarray, shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used

        nn_num : ndarray, shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in `nn_data`.

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

        y_new = np.array([y_type] * len(samples_indices), dtype=y_dtype)

        if sparse.issparse(X):
            row_indices, col_indices, samples = [], [], []
            for i, (row, col, step) in enumerate(zip(rows, cols, steps)):
                if X[row].nnz:
                    sample = self._generate_sample(X, nn_data, nn_num,
                                                   row, col, step)
                    row_indices += [i] * len(sample.indices)
                    col_indices += sample.indices.tolist()
                    samples += sample.data.tolist()
            return (sparse.csr_matrix((samples, (row_indices, col_indices)),
                                      [len(samples_indices), X.shape[1]],
                                      dtype=X.dtype),
                    y_new)
        else:
            X_new = np.zeros((n_samples, X.shape[1]), dtype=X.dtype)
            for i, (row, col, step) in enumerate(zip(rows, cols, steps)):
                X_new[i] = self._generate_sample(X, nn_data, nn_num,
                                                 row, col, step)
            return X_new, y_new

    def _generate_sample(self, X, nn_data, nn_num, row, col, step):
        """Generate a synthetic sample.

        The rule for the generation is:

        .. math:: \mathbf{s_{s}} = \mathbf{s_{i}} + \mathcal{u}(0, 1) \times (\mathbf{s_{i}} - \mathbf{s_{nn}}) \,

        where \mathbf{s_{s}} is the new synthetic samples, \mathbf{s_{i}} is the current sample,
        \mathbf{s_{nn}} is a randomly selected neighbors of \mathbf{s_{i}} and \mathcal{u}(0, 1)
        is a random number between [0, 1).

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Points from which the points will be created.

        nn_data : ndarray, shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used.

        nn_num : ndarray, shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in `nn_data`.

        row : int
            Index pointing at feature vector in X which will be used
            as a base for creating new sample.

        col : int
            Index pointing at which nearest neighbor of base feature vector
            will be used when creating new sample.

        step : float
            Step size for new sample.

        Returns
        -------
        X_new : {ndarray, sparse matrix}, shape (n_features,)
            Single synthetically generated sample.

        """
        return X[row] - step * (X[row] - nn_data[nn_num[row, col]])

    def _in_danger_noise(self, nn_estimator, samples, target_class, y,
                         kind='danger'):
        """Estimate if a set of sample are in danger or noise.

        Used by BorderlineSMOTE and SVMSMOTE.

        Parameters
        ----------
        nn_estimator : estimator
            An estimator that inherits from
            :class:`sklearn.neighbors.base.KNeighborsMixin` use to determine if
            a sample is in danger/noise.

        samples : {array-like, sparse matrix}, shape (n_samples, n_features)
            The samples to check if either they are in danger or not.

        target_class : int or str
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
        x = nn_estimator.kneighbors(samples, return_distance=False)[:, 1:]
        nn_label = (y[x] != target_class).astype(int)
        n_maj = np.sum(nn_label, axis=1)

        if kind == 'danger':
            # Samples are in danger for m/2 <= m' < m
            return np.bitwise_and(n_maj >= (nn_estimator.n_neighbors - 1) / 2,
                                  n_maj < nn_estimator.n_neighbors - 1)
        elif kind == 'noise':
            # Samples are noise for m = m'
            return n_maj == nn_estimator.n_neighbors - 1
        else:
            raise NotImplementedError

    def _fit_nn_k(self, X):
        self.nn_k_.fit(X)

    def _nn_k_neighbors(self, X):
        return self.nn_k_.kneighbors(X, return_distance=False)[:, 1:]


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring)
class BorderlineSMOTE(BaseSMOTE):
    """Over-sampling using Borderline SMOTE.

    This algorithm is a variant of the original SMOTE algorithm proposed in
    [2]_. Borderline samples will be detected and used to generate new
    synthetic samples.

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

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    m_neighbors : int or object, optional (default=10)
        If int, number of nearest neighbours to use to determine if a minority
        sample is in danger. If object, an estimator that inherits
        from :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used
        to find the m_neighbors.

    kind : str, optional (default='borderline-1')
        The type of SMOTE algorithm to use one of the following options:
        ``'borderline-1'``, ``'borderline-2'``.

    Notes
    -----
    See the original papers: [2]_ for more details.

    Supports multi-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    See also
    --------
    SMOTE : Over-sample using SMOTE.

    SVMSMOTE : Over-sample using SVM-SMOTE variant.

    ADASYN : Over-sample using ADASYN.

    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, "SMOTE:
       synthetic minority over-sampling technique," Journal of artificial
       intelligence research, 321-357, 2002.

    .. [2] H. Han, W. Wen-Yuan, M. Bing-Huan, "Borderline-SMOTE: a new
       over-sampling method in imbalanced data sets learning," Advances in
       intelligent computing, 878-887, 2005.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.over_sampling import \
BorderlineSMOTE # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> sm = BorderlineSMOTE(random_state=42)
    >>> X_res, y_res = sm.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 900, 1: 900}})

    """

    def __init__(self,
                 sampling_strategy='auto',
                 random_state=None,
                 k_neighbors=5,
                 n_jobs=1,
                 m_neighbors=10,
                 kind='borderline-1'):
        super(BorderlineSMOTE, self).__init__(
            sampling_strategy=sampling_strategy, random_state=random_state,
            k_neighbors=k_neighbors, n_jobs=n_jobs, ratio=None)
        self.m_neighbors = m_neighbors
        self.kind = kind

    def _validate_estimator(self):
        super(BorderlineSMOTE, self)._validate_estimator()
        self.nn_m_ = check_neighbors_object(
            'k_neighbors', self.k_neighbors, additional_neighbor=1)
        self.nn_m_.set_params(**{'n_jobs': self.n_jobs})
        if self.kind not in ('borderline-1', 'borderline-2'):
            raise ValueError('The possible "kind" of algorithm are '
                             '"borderline-1" and "borderline-2".'
                             'Got {} instead.'.format(self.kind))

    # FIXME: rename _sample -> _fit_resample in 0.6
    def _fit_resample(self, X, y):
        return self._sample(X, y)

    def _sample(self, X, y):
        self._validate_estimator()

        X_resampled = X.copy()
        y_resampled = y.copy()

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = safe_indexing(X, target_class_indices)

            self._fit_nn_m(X)
            danger_index = self._in_danger_noise(
                self.nn_m_, X_class, class_sample, y, kind='danger')
            if not any(danger_index):
                continue

            self._fit_nn_k(X_class)
            nns = self._nn_k_neighbors(safe_indexing(X_class, danger_index))

            # divergence between borderline-1 and borderline-2
            if self.kind == 'borderline-1':
                # Create synthetic samples for borderline points.
                X_new, y_new = self._make_samples(
                    safe_indexing(X_class, danger_index), y.dtype,
                    class_sample, X_class, nns, n_samples)
                if sparse.issparse(X_new):
                    X_resampled = sparse.vstack([X_resampled, X_new])
                else:
                    X_resampled = np.vstack((X_resampled, X_new))
                y_resampled = np.hstack((y_resampled, y_new))

            elif self.kind == 'borderline-2':
                random_state = check_random_state(self.random_state)
                fractions = random_state.beta(10, 10)

                # only minority
                X_new_1, y_new_1 = self._make_samples(
                    safe_indexing(X_class, danger_index),
                    y.dtype,
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
                    y.dtype,
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

    def _fit_nn_m(self, X):
        self.nn_m_.fit(X)


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring)
class SVMSMOTE(BaseSMOTE):
    """Over-sampling using SVM-SMOTE.

    Variant of SMOTE algorithm which use an SVM algorithm to detect sample to
    use for generating new synthetic samples as proposed in [2]_.

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

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    m_neighbors : int or object, optional (default=10)
        If int, number of nearest neighbours to use to determine if a minority
        sample is in danger. If object, an estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the m_neighbors.

    svm_estimator : object, optional (default=SVC())
        A parametrized :class:`sklearn.svm.SVC` classifier can be passed.

    out_step : float, optional (default=0.5)
        Step size when extrapolating.

    Notes
    -----
    See the original papers: [2]_ for more details.

    Supports multi-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    See also
    --------
    SMOTE : Over-sample using SMOTE.

    BorderlineSMOTE : Over-sample using Borderline-SMOTE.

    ADASYN : Over-sample using ADASYN.

    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, "SMOTE:
       synthetic minority over-sampling technique," Journal of artificial
       intelligence research, 321-357, 2002.

    .. [2] H. M. Nguyen, E. W. Cooper, K. Kamei, "Borderline over-sampling for
       imbalanced data classification," International Journal of Knowledge
       Engineering and Soft Data Paradigms, 3(1), pp.4-21, 2009.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.over_sampling import \
SVMSMOTE # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> sm = SVMSMOTE(random_state=42)
    >>> X_res, y_res = sm.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 900, 1: 900}})

    """

    def __init__(self,
                 sampling_strategy='auto',
                 random_state=None,
                 k_neighbors=5,
                 n_jobs=1,
                 m_neighbors=10,
                 svm_estimator=None,
                 out_step=0.5):
        super(SVMSMOTE, self).__init__(
            sampling_strategy=sampling_strategy, random_state=random_state,
            k_neighbors=k_neighbors, n_jobs=n_jobs, ratio=None)
        self.m_neighbors = m_neighbors
        self.svm_estimator = svm_estimator
        self.out_step = out_step

    def _validate_estimator(self):
        super(SVMSMOTE, self)._validate_estimator()
        self.nn_m_ = check_neighbors_object(
            'k_neighbors', self.k_neighbors, additional_neighbor=1)
        self.nn_m_.set_params(**{'n_jobs': self.n_jobs})

        if self.svm_estimator is None:
            self.svm_estimator_ = SVC(gamma='scale',
                                      random_state=self.random_state)
        elif isinstance(self.svm_estimator, SVC):
            self.svm_estimator_ = clone(self.svm_estimator)
        else:
            raise_isinstance_error('svm_estimator', [SVC],
                                   self.svm_estimator)

    # FIXME: rename _sample -> _fit_resample in 0.6
    def _fit_resample(self, X, y):
        return self._sample(X, y)

    def _sample(self, X, y):
        self._validate_estimator()
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

            self._fit_nn_m(X)
            noise_bool = self._in_danger_noise(
                self.nn_m_, support_vector, class_sample, y, kind='noise')
            support_vector = safe_indexing(
                support_vector, np.flatnonzero(np.logical_not(noise_bool)))
            danger_bool = self._in_danger_noise(
                self.nn_m_, support_vector, class_sample, y, kind='danger')
            safety_bool = np.logical_not(danger_bool)

            self._fit_nn_k(X_class)
            fractions = random_state.beta(10, 10)
            n_generated_samples = int(fractions * (n_samples + 1))
            if np.count_nonzero(danger_bool) > 0:
                nns = self._nn_k_neighbors(
                    safe_indexing(support_vector, np.flatnonzero(danger_bool)))

                X_new_1, y_new_1 = self._make_samples(
                    safe_indexing(support_vector, np.flatnonzero(danger_bool)),
                    y.dtype,
                    class_sample,
                    X_class,
                    nns,
                    n_generated_samples,
                    step_size=1.)

            if np.count_nonzero(safety_bool) > 0:
                nns = self._nn_k_neighbors(
                    safe_indexing(support_vector, np.flatnonzero(safety_bool)))

                X_new_2, y_new_2 = self._make_samples(
                    safe_indexing(support_vector, np.flatnonzero(safety_bool)),
                    y.dtype,
                    class_sample,
                    X_class,
                    nns,
                    n_samples - n_generated_samples,
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

    def _fit_nn_m(self, X):
        self.nn_m_.fit(X)


# FIXME: In 0.6, SMOTE should inherit only from BaseSMOTE.
@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring)
class SMOTE(SVMSMOTE, BorderlineSMOTE):
    """Class to perform over-sampling using SMOTE.

    This object is an implementation of SMOTE - Synthetic Minority
    Over-sampling Technique as presented in [1]_.

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

        .. deprecated:: 0.4
           ``m_neighbors`` is deprecated in 0.4 and will be removed in 0.6. Use
           :class:`BorderlineSMOTE` or :class:`SVMSMOTE` instead to use the
           intended algorithm.

    out_step : float, optional (default=0.5)
        Step size when extrapolating. Used with ``kind='svm'``.

        .. deprecated:: 0.4
           ``out_step`` is deprecated in 0.4 and will be removed in 0.6. Use
           :class:`SVMSMOTE` instead to use the intended algorithm.

    kind : str, optional (default='regular')
        The type of SMOTE algorithm to use one of the following options:
        ``'regular'``, ``'borderline1'``, ``'borderline2'``, ``'svm'``.

        .. deprecated:: 0.4
           ``kind`` is deprecated in 0.4 and will be removed in 0.6. Use
           :class:`BorderlineSMOTE` or :class:`SVMSMOTE` instead to use the
           intended algorithm.

    svm_estimator : object, optional (default=SVC())
        If ``kind='svm'``, a parametrized :class:`sklearn.svm.SVC`
        classifier can be passed.

        .. deprecated:: 0.4
           ``out_step`` is deprecated in 0.4 and will be removed in 0.6. Use
           :class:`SVMSMOTE` instead to use the intended algorithm.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

    ratio : str, dict, or callable
        .. deprecated:: 0.4
           Use the parameter ``sampling_strategy`` instead. It will be removed
           in 0.6.

    Notes
    -----
    See the original papers: [1]_ for more details.

    Supports multi-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    See also
    --------
    BorderlineSMOTE : Over-sample using the borderline-SMOTE variant.

    SVMSMOTE : Over-sample using the SVM-SMOTE variant.

    ADASYN : Over-sample using ADASYN.

    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, "SMOTE:
       synthetic minority over-sampling technique," Journal of artificial
       intelligence research, 321-357, 2002.

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
    >>> X_res, y_res = sm.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{0: 900, 1: 900}})

    """
    def __init__(self,
                 sampling_strategy='auto',
                 random_state=None,
                 k_neighbors=5,
                 m_neighbors='deprecated',
                 out_step='deprecated',
                 kind='deprecated',
                 svm_estimator='deprecated',
                 n_jobs=1,
                 ratio=None):
        # FIXME: in 0.6 call super()
        BaseSMOTE.__init__(self, sampling_strategy=sampling_strategy,
                           random_state=random_state, k_neighbors=k_neighbors,
                           n_jobs=n_jobs, ratio=ratio)
        self.kind = kind
        self.m_neighbors = m_neighbors
        self.out_step = out_step
        self.svm_estimator = svm_estimator
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        # FIXME: in 0.6 call super()
        BaseSMOTE._validate_estimator(self)
        # FIXME: remove in 0.6 after deprecation cycle
        if self.kind != 'deprecated' and not (self.kind == 'borderline-1' or
                                              self.kind == 'borderline-2'):
            if self.kind not in SMOTE_KIND:
                raise ValueError('Unknown kind for SMOTE algorithm.'
                                 ' Choices are {}. Got {} instead.'.format(
                                     SMOTE_KIND, self.kind))
            else:
                warnings.warn('"kind" is deprecated in 0.4 and will be '
                              'removed in 0.6. Use SMOTE, BorderlineSMOTE or '
                              'SVMSMOTE instead.', DeprecationWarning)

            if self.kind == 'borderline1' or self.kind == 'borderline2':
                self._sample = types.MethodType(BorderlineSMOTE._sample, self)
                self.kind = ('borderline-1' if self.kind == 'borderline1'
                             else 'borderline-2')

            elif self.kind == 'svm':
                self._sample = types.MethodType(SVMSMOTE._sample, self)

                if self.out_step == 'deprecated':
                    self.out_step = 0.5
                else:
                    warnings.warn('"out_step" is deprecated in 0.4 and will '
                                  'be removed in 0.6. Use SVMSMOTE class '
                                  'instead.', DeprecationWarning)

                if self.svm_estimator == 'deprecated':
                    warnings.warn('"svm_estimator" is deprecated in 0.4 and '
                                  'will be removed in 0.6. Use SVMSMOTE class '
                                  'instead.', DeprecationWarning)
                if (self.svm_estimator is None or
                        self.svm_estimator == 'deprecated'):
                    self.svm_estimator_ = SVC(gamma='scale',
                                              random_state=self.random_state)
                elif isinstance(self.svm_estimator, SVC):
                    self.svm_estimator_ = clone(self.svm_estimator)
                else:
                    raise_isinstance_error('svm_estimator', [SVC],
                                           self.svm_estimator)

            if self.kind != 'regular':
                if self.m_neighbors == 'deprecated':
                    self.m_neighbors = 10
                else:
                    warnings.warn('"m_neighbors" is deprecated in 0.4 and '
                                  'will be removed in 0.6. Use SVMSMOTE class '
                                  'or BorderlineSMOTE instead.',
                                  DeprecationWarning)

                self.nn_m_ = check_neighbors_object(
                    'm_neighbors', self.m_neighbors, additional_neighbor=1)
                self.nn_m_.set_params(**{'n_jobs': self.n_jobs})

    # FIXME: to be removed in 0.6
    def _fit_resample(self, X, y):
        self._validate_estimator()
        return self._sample(X, y)

    def _sample(self, X, y):
        # FIXME: uncomment in version 0.6
        # self._validate_estimator()

        X_resampled = X.copy()
        y_resampled = y.copy()

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = safe_indexing(X, target_class_indices)

            self._fit_nn_k(X_class)
            nns = self._nn_k_neighbors(X_class)
            X_new, y_new = self._make_samples(X_class, y.dtype, class_sample,
                                              X_class, nns, n_samples, 1.0)

            if sparse.issparse(X_new):
                X_resampled = sparse.vstack([X_resampled, X_new])
            else:
                X_resampled = np.vstack((X_resampled, X_new))
            y_resampled = np.hstack((y_resampled, y_new))

        return X_resampled, y_resampled


class SMOTENC(SMOTE):
    """Class to perform over-sampling using SMOTE-NC.

    Implementation of the Synthetic Minority Over-sampling Technique
    for Nominal and Continuous (SMOTE-NC) features. SMOTE-NC is intended
    to deal with mixed datasets of categorical and numerical data.

    SMOTE-NC requires to one-hot encode the categorical features before
    sampling, i.e. using :class:`sklearn.preprocessing.OneHotEncoder`.

    Read more in the :ref:`User Guide <smote_adasyn>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    {k_neighbors}

    {m_neighbors}

    {out_step}

    {kind}

    {svm_estimator}

    {n_jobs}

    {ratio}

    categorical_feature_indices : array-like, shape (n_categorical_features,)
        Indices to categorical feature ranges.
        Value of
        :attr:`sklearn.preprocessing.OneHotEncoder.feature_indices_`
        can be plugged directly.
        See :class:`sklearn.preprocessing.OneHotEncoder` for details.

    Attributes
    ----------
    std_median_ : float
        Median of standard deviations of continuous features.

    categorical_feature_indices_ : array of shape (n_categorical_features,)
        Indices to categorical feature ranges.
        Feature ``i`` in the original data is mapped to features
        from ``categorical_feature_indices_[i]`` to ``categorical_feature_indices_[i+1]``.

    continuous_feature_indices_ : array of shape (n_continuous_features,)
        Indices of columns with continuous features.

    Notes
    -----
    See the original paper [1]_ for more details.

    Supports mutli-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    See
    :ref:`sphx_glr_auto_examples_over-sampling_plot_comparison_over_sampling.py`,
    and :ref:`sphx_glr_auto_examples_over-sampling_plot_smote.py`.

    See also
    --------
    SMOTE : Over-sample using SMOTE.

    SVMSMOTE : Over-sample using SVM-SMOTE variant.

    ADASYN : Over-sample using ADASYN.

    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, "SMOTE:
       synthetic minority over-sampling technique," Journal of artificial
       intelligence research, 321-357, 2002.

    Examples
    --------

    >>> from collections import Counter
    >>> from numpy.random import RandomState
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.preprocessing import OneHotEncoder
    >>> from imblearn.over_sampling import \
SMOTENC # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape (%s, %s)' % X.shape)
    Original dataset shape (1000, 20)
    >>> print('Original dataset samples per class {}'.format(Counter(y)))
    Original dataset samples per class Counter({1: 900, 0: 100})
    >>> # replace two last columns with categorical features encoded as integers
    >>> X[:, -2:] = RandomState(10).randint(0, 4, size=(1000, 2))
    >>> # One-hot encode the categorical columns
    >>> encoder = OneHotEncoder(n_values=[4, 4], categorical_features=[18, 19])
    >>> X = encoder.fit_transform(X)
    >>> print('One-hot encoded dataset shape (%s, %s)' % X.shape)
    One-hot encoded dataset shape (1000, 26)
    >>> sm = SMOTENC(random_state=42, categorical_feature_indices=encoder.feature_indices_)
    >>> X_res, y_res = sm.fit_resample(X, y)
    >>> print('Resampled dataset samples per class {}'.format(Counter(y_res)))
    Resampled dataset samples per class Counter({0: 900, 1: 900})

    """

    def __init__(self,
                 sampling_strategy='auto',
                 random_state=None,
                 k_neighbors=5,
                 m_neighbors='deprecated',
                 out_step='deprecated',
                 kind='deprecated',
                 svm_estimator='deprecated',
                 n_jobs=1,
                 ratio=None,
                 categorical_feature_indices=None):
        super(SMOTENC, self).__init__(sampling_strategy=sampling_strategy,
                                      random_state=random_state,
                                      k_neighbors=k_neighbors,
                                      m_neighbors=m_neighbors,
                                      out_step=out_step,
                                      kind=kind,
                                      svm_estimator=svm_estimator,
                                      n_jobs=n_jobs,
                                      ratio=ratio)
        self.categorical_feature_indices = categorical_feature_indices

    def _fit_resample(self, X, y):
        if self.categorical_feature_indices is None:
            warnings.warn('No "categorical_feature_indices" were specified when '
                          'this instance was created. Will fall back '
                          'to normal SMOTE', RuntimeWarning)
            return super(SMOTENC, self)._fit_resample(X, y)

        feature_indices = check_array(self.categorical_feature_indices, ensure_2d=False,
                                      ensure_min_samples=2, estimator=self)
        n_features = X.shape[1]

        if np.any(feature_indices > n_features):
            raise ValueError('Indices of categorical features have to be less '
                             'than number of features in X: X.shape=(%s, %s)'
                             % X.shape)

        self.categorical_feature_indices_ = feature_indices
        self.continuous_feature_indices_ = np.setdiff1d(
            np.arange(n_features), np.arange(self.categorical_feature_indices_[0],
                                             self.categorical_feature_indices_[-1]))

        if self.continuous_feature_indices_.size == 0:
            raise ValueError('Looks like all features in X are '
                             'categorical which is not supported. '
                             'For this method to work X should have '
                             'at least 1 continuous feature.')

        if sparse.issparse(X):
            scaler = StandardScaler(with_mean=False,
                                    with_std=True,
                                    copy=False)
            scaler.fit(X.tocsc()[:, self.continuous_feature_indices_])
            self.std_median_ = np.median(np.sqrt(scaler.var_))
        else:
            std = np.std(X[:, self.continuous_feature_indices_], axis=0)
            self.std_median_ = np.median(std)

        return super(SMOTENC, self)._fit_resample(X, y)

    def _generate_sample(self, X, nn_data, nn_num, row, col, step):
        """Generate a synthetic sample.

        The rule for the generation is:

        .. math:: \mathbf{s_{s}} = \mathbf{s_{i}} + \mathcal{u}(0, 1) \times (\mathbf{s_{i}} - \mathbf{s_{nn}}) \,

        where \mathbf{s_{s}} is the new synthetic samples, \mathbf{s_{i}} is the current sample,
        \mathbf{s_{nn}} is a randomly selected neighbors of \mathbf{s_{i}} and \mathcal{u}(0, 1)
        is a random number between [0, 1).

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Points from which the points will be created.

        nn_data : ndarray, shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used.

        nn_num : ndarray, shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in `nn_data`.

        row : int
            Index pointing at feature vector in X which will be used
            as a base for creating new sample.

        col : int
            Index pointing at which nearest neighbor of base feature vector
            will be used when creating new sample.

        step : float
            Step size for new sample.

        Returns
        -------
        X_new : {ndarray, sparse matrix}, shape (n_features,)
            Single synthetically generated sample.

        """
        sample = super(SMOTENC, self)._generate_sample(X, nn_data, nn_num,
                                                       row, col, step)
        if not hasattr(self, "categorical_feature_indices_"):
            warnings.warn('No "categorical_feature_indices" were specified when '
                          'this instance was created. Will fall back '
                          'to normal SMOTE', RuntimeWarning)
            return sample

        is_sparse = sparse.issparse(nn_data)
        if is_sparse:
            nn_data = nn_data.tocsr()

        all_neighbors = nn_data[nn_num[row]]
        if is_sparse:
            all_neighbors = all_neighbors.tocsc()
            sample = sample.tolil()

        feature_idx_pairs = list(zip(self.categorical_feature_indices_[:-1],
                                     self.categorical_feature_indices_[1:]))
        for start, end in feature_idx_pairs:
            # FIXME: should sample for which neighbors were found be also
            # FIXME:  ... included when calculating most frequent nominal
            # FIXME:  ... feature values?
            nominal_values = all_neighbors[:, start:end]
            # FIXME: break ties randomly when several nominal values are
            # FIXME:  ... used by equal number of neighbors?
            most_used_value = np.argmax(np.sum(nominal_values, axis=0))
            new_nominal_value = np.zeros(nominal_values.shape[1])
            new_nominal_value[most_used_value] = 1
            if is_sparse:
                sample[:, start:end] = new_nominal_value
            else:
                sample[start:end] = new_nominal_value

        return sample.tocsr() if is_sparse else sample

    def _fit_nn_k(self, X):
        """Calls original method but on a modified copy if input."""
        return super(SMOTENC, self)._fit_nn_k(self._with_std_median(X))

    def _nn_k_neighbors(self, X):
        """Calls original method but on a modified copy if input."""
        return super(SMOTENC, self)._nn_k_neighbors(self._with_std_median(X))

    def _fit_nn_m(self, X):
        """Calls original method but on a modified copy if input."""
        return super(SMOTENC, self)._fit_nn_m(self._with_std_median(X))

    def _in_danger_noise(self, nn_estimator, samples, target_class, y,
                         kind='danger'):
        """Calls original method but on a modified copy if input."""
        return super(SMOTENC, self)._in_danger_noise(nn_estimator,
                                                     self._with_std_median(samples),
                                                     target_class, y, kind=kind)

    def _with_std_median(self, X):
        """
        Given that all categorical features are assumed to be one-hot encoded,
        their values are either 0 or 1. We replace values in original input
        which are equal to 1 with calculated median of standard deviations
        divided by 2. It will ensure that whenever distance is calculated
        between two feature vectors, the difference of two different categorical
        features will always equal to median standard deviation.
        """
        if not hasattr(self, "categorical_feature_indices_"):
            warnings.warn('No "categorical_feature_indices" were specified when '
                          'this instance was created. Will fallback '
                          'to normal SMOTE', RuntimeWarning)
            return X

        X_copy = X.copy().tolil() if sparse.issparse(X) else X.copy()
        start = self.categorical_feature_indices_[0]
        end = self.categorical_feature_indices_[-1]
        mask = X_copy[:, start:end] == 1
        X_copy[:, start:end][mask] = self.std_median_ / 2.0
        return X_copy.tocoo() if sparse.issparse(X_copy) else X_copy
