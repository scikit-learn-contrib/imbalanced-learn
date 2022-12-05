"""Base class and original SMOTE methods for over-sampling"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Fernando Nogueira
#          Christos Aridas
#          Dzianis Dudnik
# License: MIT

import math
import numbers
import warnings
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.utils import _safe_indexing, check_array, check_random_state
from sklearn.utils.sparsefuncs_fast import (
    csc_mean_variance_axis0,
    csr_mean_variance_axis0,
)

from ...metrics.pairwise import ValueDifferenceMetric
from ...utils import Substitution, check_neighbors_object, check_target_type
from ...utils._docstring import _n_jobs_docstring, _random_state_docstring
from ...utils._param_validation import HasMethods, Interval
from ...utils.fixes import _mode
from ..base import BaseOverSampler


class BaseSMOTE(BaseOverSampler):
    """Base class for the different SMOTE algorithms."""

    _parameter_constraints: dict = {
        **BaseOverSampler._parameter_constraints,
        "k_neighbors": [
            Interval(numbers.Integral, 1, None, closed="left"),
            HasMethods(["kneighbors", "kneighbors_graph"]),
        ],
        "n_jobs": [numbers.Integral, None],
    }

    def __init__(
        self,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Check the NN estimators shared across the different SMOTE
        algorithms.
        """
        self.nn_k_ = check_neighbors_object(
            "k_neighbors", self.k_neighbors, additional_neighbor=1
        )

    def _make_samples(
        self, X, y_dtype, y_type, nn_data, nn_num, n_samples, step_size=1.0
    ):
        """A support function that returns artificial samples constructed along
        the line connecting nearest neighbours.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Points from which the points will be created.

        y_dtype : dtype
            The data type of the targets.

        y_type : str or int
            The minority target value, just so the function can return the
            target values for the synthetic variables with correct length in
            a clear format.

        nn_data : ndarray of shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used

        nn_num : ndarray of shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in `nn_data`.

        n_samples : int
            The number of samples to generate.

        step_size : float, default=1.0
            The step size to create samples.

        Returns
        -------
        X_new : {ndarray, sparse matrix} of shape (n_samples_new, n_features)
            Synthetically generated samples.

        y_new : ndarray of shape (n_samples_new,)
            Target values for synthetic samples.
        """
        random_state = check_random_state(self.random_state)
        samples_indices = random_state.randint(low=0, high=nn_num.size, size=n_samples)

        # np.newaxis for backwards compatability with random_state
        steps = step_size * random_state.uniform(size=n_samples)[:, np.newaxis]
        rows = np.floor_divide(samples_indices, nn_num.shape[1])
        cols = np.mod(samples_indices, nn_num.shape[1])

        X_new = self._generate_samples(X, nn_data, nn_num, rows, cols, steps)
        y_new = np.full(n_samples, fill_value=y_type, dtype=y_dtype)
        return X_new, y_new

    def _generate_samples(self, X, nn_data, nn_num, rows, cols, steps):
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
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Points from which the points will be created.

        nn_data : ndarray of shape (n_samples_all, n_features)
            Data set carrying all the neighbours to be used.

        nn_num : ndarray of shape (n_samples_all, k_nearest_neighbours)
            The nearest neighbours of each sample in `nn_data`.

        rows : ndarray of shape (n_samples,), dtype=int
            Indices pointing at feature vector in X which will be used
            as a base for creating new samples.

        cols : ndarray of shape (n_samples,), dtype=int
            Indices pointing at which nearest neighbor of base feature vector
            will be used when creating new samples.

        steps : ndarray of shape (n_samples,), dtype=float
            Step sizes for new samples.

        Returns
        -------
        X_new : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Synthetically generated samples.
        """
        diffs = nn_data[nn_num[rows, cols]] - X[rows]

        if sparse.issparse(X):
            sparse_func = type(X).__name__
            steps = getattr(sparse, sparse_func)(steps)
            X_new = X[rows] + steps.multiply(diffs)
        else:
            X_new = X[rows] + steps * diffs

        return X_new.astype(X.dtype)

    def _in_danger_noise(self, nn_estimator, samples, target_class, y, kind="danger"):
        """Estimate if a set of sample are in danger or noise.

        Used by BorderlineSMOTE and SVMSMOTE.

        Parameters
        ----------
        nn_estimator : estimator object
            An estimator that inherits from
            :class:`~sklearn.neighbors.base.KNeighborsMixin` use to determine
            if a sample is in danger/noise.

        samples : {array-like, sparse matrix} of shape (n_samples, n_features)
            The samples to check if either they are in danger or not.

        target_class : int or str
            The target corresponding class being over-sampled.

        y : array-like of shape (n_samples,)
            The true label in order to check the neighbour labels.

        kind : {'danger', 'noise'}, default='danger'
            The type of classification to use. Can be either:

            - If 'danger', check if samples are in danger,
            - If 'noise', check if samples are noise.

        Returns
        -------
        output : ndarray of shape (n_samples,)
            A boolean array where True refer to samples in danger or noise.
        """
        x = nn_estimator.kneighbors(samples, return_distance=False)[:, 1:]
        nn_label = (y[x] != target_class).astype(int)
        n_maj = np.sum(nn_label, axis=1)

        if kind == "danger":
            # Samples are in danger for m/2 <= m' < m
            return np.bitwise_and(
                n_maj >= (nn_estimator.n_neighbors - 1) / 2,
                n_maj < nn_estimator.n_neighbors - 1,
            )
        else:  # kind == "noise":
            # Samples are noise for m = m'
            return n_maj == nn_estimator.n_neighbors - 1


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class SMOTE(BaseSMOTE):
    """Class to perform over-sampling using SMOTE.

    This object is an implementation of SMOTE - Synthetic Minority
    Over-sampling Technique as presented in [1]_.

    Read more in the :ref:`User Guide <smote_adasyn>`.

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    k_neighbors : int or object, default=5
        The nearest neighbors used to define the neighborhood of samples to use
        to generate the synthetic samples. You can pass:

        - an `int` corresponding to the number of neighbors to use. A
          `~sklearn.neighbors.NearestNeighbors` instance will be fitted in this
          case.
        - an instance of a compatible nearest neighbors algorithm that should
          implement both methods `kneighbors` and `kneighbors_graph`. For
          instance, it could correspond to a
          :class:`~sklearn.neighbors.NearestNeighbors` but could be extended to
          any compatible class.

    {n_jobs}

        .. deprecated:: 0.10
           `n_jobs` has been deprecated in 0.10 and will be removed in 0.12.
           It was previously used to set `n_jobs` of nearest neighbors
           algorithm. From now on, you can pass an estimator where `n_jobs` is
           already set instead.

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    nn_k_ : estimator object
        Validated k-nearest neighbours created from the `k_neighbors` parameter.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    SMOTENC : Over-sample using SMOTE for continuous and categorical features.

    SMOTEN : Over-sample using the SMOTE variant specifically for categorical
        features only.

    BorderlineSMOTE : Over-sample using the borderline-SMOTE variant.

    SVMSMOTE : Over-sample using the SVM-SMOTE variant.

    ADASYN : Over-sample using ADASYN.

    KMeansSMOTE : Over-sample applying a clustering before to oversample using
        SMOTE.

    Notes
    -----
    See the original papers: [1]_ for more details.

    Supports multi-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, "SMOTE:
       synthetic minority over-sampling technique," Journal of artificial
       intelligence research, 321-357, 2002.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.over_sampling import SMOTE
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

    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )

    def _fit_resample(self, X, y):
        # FIXME: to be removed in 0.12
        if self.n_jobs is not None:
            warnings.warn(
                "The parameter `n_jobs` has been deprecated in 0.10 and will be "
                "removed in 0.12. You can pass an nearest neighbors estimator where "
                "`n_jobs` is already set instead.",
                FutureWarning,
            )

        self._validate_estimator()

        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = _safe_indexing(X, target_class_indices)

            self.nn_k_.fit(X_class)
            nns = self.nn_k_.kneighbors(X_class, return_distance=False)[:, 1:]
            X_new, y_new = self._make_samples(
                X_class, y.dtype, class_sample, X_class, nns, n_samples, 1.0
            )
            X_resampled.append(X_new)
            y_resampled.append(y_new)

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:
            X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        return X_resampled, y_resampled


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class SMOTENC(SMOTE):
    """Synthetic Minority Over-sampling Technique for Nominal and Continuous.

    Unlike :class:`SMOTE`, SMOTE-NC for dataset containing numerical and
    categorical features. However, it is not designed to work with only
    categorical features.

    Read more in the :ref:`User Guide <smote_adasyn>`.

    .. versionadded:: 0.4

    Parameters
    ----------
    categorical_features : array-like of shape (n_cat_features,) or (n_features,)
        Specified which features are categorical. Can either be:

        - array of indices specifying the categorical features;
        - mask array of shape (n_features, ) and ``bool`` dtype for which
          ``True`` indicates the categorical features.

    {sampling_strategy}

    {random_state}

    k_neighbors : int or object, default=5
        The nearest neighbors used to define the neighborhood of samples to use
        to generate the synthetic samples. You can pass:

        - an `int` corresponding to the number of neighbors to use. A
          `~sklearn.neighbors.NearestNeighbors` instance will be fitted in this
          case.
        - an instance of a compatible nearest neighbors algorithm that should
          implement both methods `kneighbors` and `kneighbors_graph`. For
          instance, it could correspond to a
          :class:`~sklearn.neighbors.NearestNeighbors` but could be extended to
          any compatible class.

    {n_jobs}

        .. deprecated:: 0.10
           `n_jobs` has been deprecated in 0.10 and will be removed in 0.12.
           It was previously used to set `n_jobs` of nearest neighbors
           algorithm. From now on, you can pass an estimator where `n_jobs` is
           already set instead.

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    nn_k_ : estimator object
        Validated k-nearest neighbours created from the `k_neighbors` parameter.

    ohe_ : :class:`~sklearn.preprocessing.OneHotEncoder`
        The one-hot encoder used to encode the categorical features.

    categorical_features_ : ndarray of shape (n_cat_features,), dtype=np.int64
        Indices of the categorical features.

    continuous_features_ : ndarray of shape (n_cont_features,), dtype=np.int64
        Indices of the continuous features.

    median_std_ : float
        Median of the standard deviation of the continuous features.

    n_features_ : int
        Number of features observed at `fit`.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    SMOTE : Over-sample using SMOTE.

    SMOTEN : Over-sample using the SMOTE variant specifically for categorical
        features only.

    SVMSMOTE : Over-sample using SVM-SMOTE variant.

    BorderlineSMOTE : Over-sample using Borderline-SMOTE variant.

    ADASYN : Over-sample using ADASYN.

    KMeansSMOTE : Over-sample applying a clustering before to oversample using
        SMOTE.

    Notes
    -----
    See the original paper [1]_ for more details.

    Supports multi-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    See
    :ref:`sphx_glr_auto_examples_over-sampling_plot_comparison_over_sampling.py`,
    and :ref:`sphx_glr_auto_examples_over-sampling_plot_illustration_generation_sample.py`.  # noqa

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
    >>> print(f'Original dataset shape {{X.shape}}')
    Original dataset shape (1000, 20)
    >>> print(f'Original dataset samples per class {{Counter(y)}}')
    Original dataset samples per class Counter({{1: 900, 0: 100}})
    >>> # simulate the 2 last columns to be categorical features
    >>> X[:, -2:] = RandomState(10).randint(0, 4, size=(1000, 2))
    >>> sm = SMOTENC(random_state=42, categorical_features=[18, 19])
    >>> X_res, y_res = sm.fit_resample(X, y)
    >>> print(f'Resampled dataset samples per class {{Counter(y_res)}}')
    Resampled dataset samples per class Counter({{0: 900, 1: 900}})
    """

    _required_parameters = ["categorical_features"]

    _parameter_constraints: dict = {
        **SMOTE._parameter_constraints,
        "categorical_features": ["array-like"],
    }

    def __init__(
        self,
        categorical_features,
        *,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        n_jobs=None,
    ):
        super().__init__(
            sampling_strategy=sampling_strategy,
            random_state=random_state,
            k_neighbors=k_neighbors,
            n_jobs=n_jobs,
        )
        self.categorical_features = categorical_features

    def _check_X_y(self, X, y):
        """Overwrite the checking to let pass some string for categorical
        features.
        """
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = self._validate_data(
            X, y, reset=True, dtype=None, accept_sparse=["csr", "csc"]
        )
        return X, y, binarize_y

    def _validate_estimator(self):
        super()._validate_estimator()
        categorical_features = np.asarray(self.categorical_features)
        if categorical_features.dtype.name == "bool":
            self.categorical_features_ = np.flatnonzero(categorical_features)
        else:
            if any(
                [cat not in np.arange(self.n_features_) for cat in categorical_features]
            ):
                raise ValueError(
                    f"Some of the categorical indices are out of range. Indices"
                    f" should be between 0 and {self.n_features_ - 1}"
                )
            self.categorical_features_ = categorical_features
        self.continuous_features_ = np.setdiff1d(
            np.arange(self.n_features_), self.categorical_features_
        )

        if self.categorical_features_.size == self.n_features_in_:
            raise ValueError(
                "SMOTE-NC is not designed to work only with categorical "
                "features. It requires some numerical features."
            )

    def _fit_resample(self, X, y):
        # FIXME: to be removed in 0.12
        if self.n_jobs is not None:
            warnings.warn(
                "The parameter `n_jobs` has been deprecated in 0.10 and will be "
                "removed in 0.12. You can pass an nearest neighbors estimator where "
                "`n_jobs` is already set instead.",
                FutureWarning,
            )

        self.n_features_ = X.shape[1]
        self._validate_estimator()

        # compute the median of the standard deviation of the minority class
        target_stats = Counter(y)
        class_minority = min(target_stats, key=target_stats.get)

        X_continuous = X[:, self.continuous_features_]
        X_continuous = check_array(X_continuous, accept_sparse=["csr", "csc"])
        X_minority = _safe_indexing(X_continuous, np.flatnonzero(y == class_minority))

        if sparse.issparse(X):
            if X.format == "csr":
                _, var = csr_mean_variance_axis0(X_minority)
            else:
                _, var = csc_mean_variance_axis0(X_minority)
        else:
            var = X_minority.var(axis=0)
        self.median_std_ = np.median(np.sqrt(var))

        X_categorical = X[:, self.categorical_features_]
        if X_continuous.dtype.name != "object":
            dtype_ohe = X_continuous.dtype
        else:
            dtype_ohe = np.float64

        self.ohe_ = OneHotEncoder(handle_unknown="ignore", dtype=dtype_ohe)
        if hasattr(self.ohe_, "sparse_output"):
            # scikit-learn >= 1.2
            self.ohe_.set_params(sparse_output=True)
        else:
            self.ohe_.set_params(sparse=True)

        # the input of the OneHotEncoder needs to be dense
        X_ohe = self.ohe_.fit_transform(
            X_categorical.toarray() if sparse.issparse(X_categorical) else X_categorical
        )

        # we can replace the 1 entries of the categorical features with the
        # median of the standard deviation. It will ensure that whenever
        # distance is computed between 2 samples, the difference will be equal
        # to the median of the standard deviation as in the original paper.

        # In the edge case where the median of the std is equal to 0, the 1s
        # entries will be also nullified. In this case, we store the original
        # categorical encoding which will be later used for inversing the OHE
        if math.isclose(self.median_std_, 0):
            self._X_categorical_minority_encoded = _safe_indexing(
                X_ohe.toarray(), np.flatnonzero(y == class_minority)
            )

        X_ohe.data = np.ones_like(X_ohe.data, dtype=X_ohe.dtype) * self.median_std_ / 2
        X_encoded = sparse.hstack((X_continuous, X_ohe), format="csr")

        X_resampled, y_resampled = super()._fit_resample(X_encoded, y)

        # reverse the encoding of the categorical features
        X_res_cat = X_resampled[:, self.continuous_features_.size :]
        X_res_cat.data = np.ones_like(X_res_cat.data)
        X_res_cat_dec = self.ohe_.inverse_transform(X_res_cat)

        if sparse.issparse(X):
            X_resampled = sparse.hstack(
                (
                    X_resampled[:, : self.continuous_features_.size],
                    X_res_cat_dec,
                ),
                format="csr",
            )
        else:
            X_resampled = np.hstack(
                (
                    X_resampled[:, : self.continuous_features_.size].toarray(),
                    X_res_cat_dec,
                )
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

    def _generate_samples(self, X, nn_data, nn_num, rows, cols, steps):
        """Generate a synthetic sample with an additional steps for the
        categorical features.

        Each new sample is generated the same way than in SMOTE. However, the
        categorical features are mapped to the most frequent nearest neighbors
        of the majority class.
        """
        rng = check_random_state(self.random_state)
        X_new = super()._generate_samples(X, nn_data, nn_num, rows, cols, steps)
        # change in sparsity structure more efficient with LIL than CSR
        X_new = X_new.tolil() if sparse.issparse(X_new) else X_new

        # convert to dense array since scipy.sparse doesn't handle 3D
        nn_data = nn_data.toarray() if sparse.issparse(nn_data) else nn_data

        # In the case that the median std was equal to zeros, we have to
        # create non-null entry based on the encoded of OHE
        if math.isclose(self.median_std_, 0):
            nn_data[
                :, self.continuous_features_.size :
            ] = self._X_categorical_minority_encoded

        all_neighbors = nn_data[nn_num[rows]]

        categories_size = [self.continuous_features_.size] + [
            cat.size for cat in self.ohe_.categories_
        ]

        for start_idx, end_idx in zip(
            np.cumsum(categories_size)[:-1], np.cumsum(categories_size)[1:]
        ):
            col_maxs = all_neighbors[:, :, start_idx:end_idx].sum(axis=1)
            # tie breaking argmax
            is_max = np.isclose(col_maxs, col_maxs.max(axis=1, keepdims=True))
            max_idxs = rng.permutation(np.argwhere(is_max))
            xs, idx_sels = np.unique(max_idxs[:, 0], return_index=True)
            col_sels = max_idxs[idx_sels, 1]

            ys = start_idx + col_sels
            X_new[:, start_idx:end_idx] = 0
            X_new[xs, ys] = 1

        return X_new


@Substitution(
    sampling_strategy=BaseOverSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class SMOTEN(SMOTE):
    """Synthetic Minority Over-sampling Technique for Nominal.

    This method is referred as SMOTEN in [1]_. It expects that the data to
    resample are only made of categorical features.

    Read more in the :ref:`User Guide <smote_adasyn>`.

    .. versionadded:: 0.8

    Parameters
    ----------
    {sampling_strategy}

    {random_state}

    k_neighbors : int or object, default=5
        The nearest neighbors used to define the neighborhood of samples to use
        to generate the synthetic samples. You can pass:

        - an `int` corresponding to the number of neighbors to use. A
          `~sklearn.neighbors.NearestNeighbors` instance will be fitted in this
          case.
        - an instance of a compatible nearest neighbors algorithm that should
          implement both methods `kneighbors` and `kneighbors_graph`. For
          instance, it could correspond to a
          :class:`~sklearn.neighbors.NearestNeighbors` but could be extended to
          any compatible class.

    {n_jobs}

        .. deprecated:: 0.10
           `n_jobs` has been deprecated in 0.10 and will be removed in 0.12.
           It was previously used to set `n_jobs` of nearest neighbors
           algorithm. From now on, you can pass an estimator where `n_jobs` is
           already set instead.

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        corresponds to the class labels from which to sample and the values
        are the number of samples to sample.

    nn_k_ : estimator object
        Validated k-nearest neighbours created from the `k_neighbors` parameter.

    n_features_in_ : int
        Number of features in the input dataset.

        .. versionadded:: 0.9

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

        .. versionadded:: 0.10

    See Also
    --------
    SMOTE : Over-sample using SMOTE.

    SMOTENC : Over-sample using SMOTE for continuous and categorical features.

    BorderlineSMOTE : Over-sample using the borderline-SMOTE variant.

    SVMSMOTE : Over-sample using the SVM-SMOTE variant.

    ADASYN : Over-sample using ADASYN.

    KMeansSMOTE : Over-sample applying a clustering before to oversample using
        SMOTE.

    Notes
    -----
    See the original papers: [1]_ for more details.

    Supports multi-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    References
    ----------
    .. [1] N. V. Chawla, K. W. Bowyer, L. O.Hall, W. P. Kegelmeyer, "SMOTE:
       synthetic minority over-sampling technique," Journal of artificial
       intelligence research, 321-357, 2002.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array(["A"] * 10 + ["B"] * 20 + ["C"] * 30, dtype=object).reshape(-1, 1)
    >>> y = np.array([0] * 20 + [1] * 40, dtype=np.int32)
    >>> from collections import Counter
    >>> print(f"Original class counts: {{Counter(y)}}")
    Original class counts: Counter({{1: 40, 0: 20}})
    >>> from imblearn.over_sampling import SMOTEN
    >>> sampler = SMOTEN(random_state=0)
    >>> X_res, y_res = sampler.fit_resample(X, y)
    >>> print(f"Class counts after resampling {{Counter(y_res)}}")
    Class counts after resampling Counter({{0: 40, 1: 40}})
    """

    def _check_X_y(self, X, y):
        """Check should accept strings and not sparse matrices."""
        y, binarize_y = check_target_type(y, indicate_one_vs_all=True)
        X, y = self._validate_data(
            X,
            y,
            reset=True,
            dtype=None,
            accept_sparse=False,
        )
        return X, y, binarize_y

    def _validate_estimator(self):
        """Force to use precomputed distance matrix."""
        super()._validate_estimator()
        self.nn_k_.set_params(metric="precomputed")

    def _make_samples(self, X_class, klass, y_dtype, nn_indices, n_samples):
        random_state = check_random_state(self.random_state)
        # generate sample indices that will be used to generate new samples
        samples_indices = random_state.choice(
            np.arange(X_class.shape[0]), size=n_samples, replace=True
        )
        # for each drawn samples, select its k-neighbors and generate a sample
        # where for each feature individually, each category generated is the
        # most common category
        X_new = np.squeeze(
            _mode(X_class[nn_indices[samples_indices]], axis=1).mode, axis=1
        )
        y_new = np.full(n_samples, fill_value=klass, dtype=y_dtype)
        return X_new, y_new

    def _fit_resample(self, X, y):
        # FIXME: to be removed in 0.12
        if self.n_jobs is not None:
            warnings.warn(
                "The parameter `n_jobs` has been deprecated in 0.10 and will be "
                "removed in 0.12. You can pass an nearest neighbors estimator where "
                "`n_jobs` is already set instead.",
                FutureWarning,
            )

        self._validate_estimator()

        X_resampled = [X.copy()]
        y_resampled = [y.copy()]

        encoder = OrdinalEncoder(dtype=np.int32)
        X_encoded = encoder.fit_transform(X)

        vdm = ValueDifferenceMetric(
            n_categories=[len(cat) for cat in encoder.categories_]
        ).fit(X_encoded, y)

        for class_sample, n_samples in self.sampling_strategy_.items():
            if n_samples == 0:
                continue
            target_class_indices = np.flatnonzero(y == class_sample)
            X_class = _safe_indexing(X_encoded, target_class_indices)

            X_class_dist = vdm.pairwise(X_class)
            self.nn_k_.fit(X_class_dist)
            # the kneigbors search will include the sample itself which is
            # expected from the original algorithm
            nn_indices = self.nn_k_.kneighbors(X_class_dist, return_distance=False)
            X_new, y_new = self._make_samples(
                X_class, class_sample, y.dtype, nn_indices, n_samples
            )

            X_new = encoder.inverse_transform(X_new)
            X_resampled.append(X_new)
            y_resampled.append(y_new)

        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        return X_resampled, y_resampled

    def _more_tags(self):
        return {"X_types": ["2darray", "dataframe", "string"]}
