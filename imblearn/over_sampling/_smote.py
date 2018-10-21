"""Class to perform over-sampling using SMOTE."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Fernando Nogueira
#          Christos Aridas
#          Dzianis Dudnik
# License: MIT

from __future__ import division

import types
import warnings
from collections import Counter

import numpy as np
from scipy import sparse

from sklearn.base import clone
from sklearn.preprocessing import OneHotEncoder
from sklearn.svm import SVC
from sklearn.utils import check_random_state
from sklearn.utils import safe_indexing
from sklearn.utils import check_array
from sklearn.utils import check_X_y
from sklearn.utils.sparsefuncs_fast import csr_mean_variance_axis0
from sklearn.utils.sparsefuncs_fast import csc_mean_variance_axis0

from .base import BaseOverSampler
from ..exceptions import raise_isinstance_error
from ..utils import check_neighbors_object
from ..utils import check_target_type
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
        r"""Generate a synthetic sample.

        The rule for the generation is:

        .. math::
           \mathbf{s_{s}} = \mathbf{s_{i}} + \mathcal{u}(0, 1) \times
           (\mathbf{s_{i}} - \mathbf{s_{nn}}) \,

        where \mathbf{s_{s}} is the new synthetic samples, \mathbf{s_{i}} is
        the current sample, \mathbf{s_{nn}} is a randomly selected neighbors of
        \mathbf{s_{i}} and \mathcal{u}(0, 1) is a random number between [0, 1).

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

    SMOTENC : Over-sample using SMOTE for continuous and categorical features.

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

            self.nn_m_.fit(X)
            danger_index = self._in_danger_noise(
                self.nn_m_, X_class, class_sample, y, kind='danger')
            if not any(danger_index):
                continue

            self.nn_k_.fit(X_class)
            nns = self.nn_k_.kneighbors(safe_indexing(X_class, danger_index),
                                        return_distance=False)[:, 1:]

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

    SMOTENC : Over-sample using SMOTE for continuous and categorical features.

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

            self.nn_m_.fit(X)
            noise_bool = self._in_danger_noise(
                self.nn_m_, support_vector, class_sample, y, kind='noise')
            support_vector = safe_indexing(
                support_vector, np.flatnonzero(np.logical_not(noise_bool)))
            danger_bool = self._in_danger_noise(
                self.nn_m_, support_vector, class_sample, y, kind='danger')
            safety_bool = np.logical_not(danger_bool)

            self.nn_k_.fit(X_class)
            fractions = random_state.beta(10, 10)
            n_generated_samples = int(fractions * (n_samples + 1))
            if np.count_nonzero(danger_bool) > 0:
                nns = self.nn_k_.kneighbors(
                    safe_indexing(support_vector, np.flatnonzero(danger_bool)),
                    return_distance=False)[:, 1:]

                X_new_1, y_new_1 = self._make_samples(
                    safe_indexing(support_vector, np.flatnonzero(danger_bool)),
                    y.dtype,
                    class_sample,
                    X_class,
                    nns,
                    n_generated_samples,
                    step_size=1.)

            if np.count_nonzero(safety_bool) > 0:
                nns = self.nn_k_.kneighbors(
                    safe_indexing(support_vector, np.flatnonzero(safety_bool)),
                    return_distance=False)[:, 1:]

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
    SMOTENC : Over-sample using SMOTE for continuous and categorical features.

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

            self.nn_k_.fit(X_class)
            nns = self.nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]
            X_new, y_new = self._make_samples(X_class, y.dtype, class_sample,
                                              X_class, nns, n_samples, 1.0)

            if sparse.issparse(X_new):
                X_resampled = sparse.vstack([X_resampled, X_new])
            else:
                X_resampled = np.vstack((X_resampled, X_new))
            y_resampled = np.hstack((y_resampled, y_new))

        return X_resampled, y_resampled


class SMOTENC(SMOTE):
    """Synthetic Minority Over-sampling Technique for Nominal and Continuous
    (SMOTE-NC).

    Unlike :class:`SMOTE`, SMOTE-NC for dataset containing continuous and
    categorical features.

    Read more in the :ref:`User Guide <smote_adasyn>`.

    Parameters
    ----------
    categorical_features : ndarray, shape (n_cat_features,) or (n_features,)
        Specified which features are categorical. Can either be:

        - array of indices specifying the categorical features;
        - mask array of shape (n_features, ) and ``bool`` dtype for which
          ``True`` indicates the categorical features.

    {sampling_strategy}

    {random_state}

    k_neighbors : int or object, optional (default=5)
        If ``int``, number of nearest neighbours to used to construct synthetic
        samples.  If object, an estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.

    n_jobs : int, optional (default=1)
        The number of threads to open if possible.

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

    BorderlineSMOTE : Over-sample using Borderline-SMOTE variant.

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
    >>> from imblearn.over_sampling import SMOTENC
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape (%s, %s)' % X.shape)
    Original dataset shape (1000, 20)
    >>> print('Original dataset samples per class {}'.format(Counter(y)))
    Original dataset samples per class Counter({1: 900, 0: 100})
    >>> # simulate the 2 last columns to be categorical features
    >>> X[:, -2:] = RandomState(10).randint(0, 4, size=(1000, 2))
    >>> sm = SMOTENC(random_state=42, categorical_features=[18, 19])
    >>> X_res, y_res = sm.fit_resample(X, y)
    >>> print('Resampled dataset samples per class {}'.format(Counter(y_res)))
    Resampled dataset samples per class Counter({0: 900, 1: 900})

    """

    def __init__(self, categorical_features, sampling_strategy='auto',
                 random_state=None, k_neighbors=5, n_jobs=1):
        super(SMOTENC, self).__init__(sampling_strategy=sampling_strategy,
                                      random_state=random_state,
                                      k_neighbors=k_neighbors,
                                      ratio=None)
        self.categorical_features = categorical_features

    @staticmethod
    def _check_X_y(X, y):
        """Overwrite the checking to let pass some string for categorical
        features.
        """
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc'], dtype=None)
        return X, y, binarize_y

    def _validate_estimator(self):
        super(SMOTENC, self)._validate_estimator()
        categorical_features = np.asarray(self.categorical_features)
        if categorical_features.dtype.name == 'bool':
            self.categorical_features_ = np.flatnonzero(categorical_features)
        else:
            if any([cat not in np.arange(self.n_features_)
                    for cat in categorical_features]):
                raise ValueError(
                    'Some of the categorical indices are out of range. Indices'
                    ' should be between 0 and {}'.format(self.n_features_))
            self.categorical_features_ = categorical_features
        self.continuous_features_ = np.setdiff1d(np.arange(self.n_features_),
                                                 self.categorical_features_)

    def _fit_resample(self, X, y):
        self.n_features_ = X.shape[1]
        self._validate_estimator()

        # compute the median of the standard deviation of the minority class
        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)

        X_continuous = X[:, self.continuous_features_]
        X_continuous = check_array(X_continuous, accept_sparse=['csr', 'csc'])
        X_minority = safe_indexing(X_continuous,
                                   np.flatnonzero(y == class_minority))

        if sparse.issparse(X):
            if X.format == 'csr':
                _, var = csr_mean_variance_axis0(X_minority)
            else:
                _, var = csc_mean_variance_axis0(X_minority)
        else:
            var = X_minority.var(axis=0)
        self.median_std_ = np.median(np.sqrt(var))

        X_categorical = X[:, self.categorical_features_]
        if X_continuous.dtype.name != 'object':
            dtype_ohe = X_continuous.dtype
        else:
            dtype_ohe = np.float64
        self.ohe_ = OneHotEncoder(sparse=True, handle_unknown='ignore',
                                  dtype=dtype_ohe)
        # the input of the OneHotEncoder needs to be dense
        X_ohe = self.ohe_.fit_transform(
            X_categorical.toarray() if sparse.issparse(X_categorical)
            else X_categorical)

        # we can replace the 1 entries of the categorical features with the
        # median of the standard deviation. It will ensure that whenever
        # distance is computed between 2 samples, the difference will be equal
        # to the median of the standard deviation as in the original paper.
        X_ohe.data = (np.ones_like(X_ohe.data, dtype=X_ohe.dtype) *
                      self.median_std_ / 2)
        X_encoded = sparse.hstack((X_continuous, X_ohe), format='csr')

        X_resampled, y_resampled = super(SMOTENC, self)._fit_resample(
            X_encoded, y)

        # reverse the encoding of the categorical features
        X_res_cat = X_resampled[:, self.continuous_features_.size:]
        X_res_cat.data = np.ones_like(X_res_cat.data)
        X_res_cat_dec = self.ohe_.inverse_transform(X_res_cat.toarray())

        if sparse.issparse(X):
            X_resampled = sparse.hstack(
                (X_resampled[:, :self.continuous_features_.size],
                 X_res_cat_dec), format='csr'
            )
        else:
            X_resampled = np.hstack(
                (X_resampled[:, :self.continuous_features_.size].toarray(),
                 X_res_cat_dec)
            )

        indices_reordered = np.argsort(
            np.hstack((self.continuous_features_, self.categorical_features_))
        )
        if sparse.issparse(X_resampled):
            # the matrix is supposed to be in the CSR format after the stacking
            col_indices = X_resampled.indices.copy()
            for idx, col_idx in enumerate(indices_reordered):
                mask = X_resampled.indices == col_idx
                col_indices[mask] = idx
            X_resampled.indices = col_indices
        else:
            X_resampled = X_resampled[:, indices_reordered]

        return X_resampled, y_resampled

    def _generate_sample(self, X, nn_data, nn_num, row, col, step):
        """Generate a synthetic sample with an additional steps for the
        categorical features.

        Each new sample is generated the same way than in SMOTE. However, the
        categorical features are mapped to the most frequent nearest neighbors
        of the majority class.
        """
        rng = check_random_state(self.random_state)
        sample = super(SMOTENC, self)._generate_sample(X, nn_data, nn_num,
                                                       row, col, step)
        # To avoid conversion and since there is only few samples used, we
        # convert those samples to dense array.
        sample = (sample.toarray().squeeze()
                  if sparse.issparse(sample) else sample)
        all_neighbors = nn_data[nn_num[row]]
        all_neighbors = (all_neighbors.toarray()
                         if sparse.issparse(all_neighbors) else all_neighbors)

        categories_size = ([self.continuous_features_.size] +
                           [cat.size for cat in self.ohe_.categories_])
        for start_idx, end_idx in zip(np.cumsum(categories_size)[:-1],
                                      np.cumsum(categories_size)[1:]):
            col_max = all_neighbors[:, start_idx:end_idx].sum(axis=0)
            # tie breaking argmax
            col_sel = rng.choice(col_max == col_max.max())
            sample[start_idx:end_idx] = 0
            sample[start_idx + col_sel] = 1

        return sparse.csr_matrix(sample) if sparse.issparse(X) else sample
