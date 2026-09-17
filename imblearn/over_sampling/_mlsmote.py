"""Class to perform over-sampling using MLSMOTE."""

# Authors: Omar Alaaeldein
# License: MIT

import numbers

import numpy as np
from scipy import sparse
from sklearn.utils import check_consistent_length, check_random_state
from sklearn.utils._param_validation import Interval, StrOptions
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn_compat.base import _fit_context
from sklearn_compat.utils.validation import check_array, validate_data

from imblearn.over_sampling.base import BaseOverSampler
from imblearn.utils import Substitution
from imblearn.utils._docstring import _random_state_docstring
from imblearn.utils._validation import ArraysTransformer


def _most_frequent(values):
    """Return the most frequent value, breaking ties by first occurrence."""
    uniques, counts = np.unique(values, return_counts=True)
    return uniques[int(np.argmax(counts))]


def _positive_proba_per_value(x_col, y_col):
    """Estimate `P(y=1 | x_col=value)` for each sample in `x_col`.

    Parameters
    ----------
    x_col : ndarray of shape (n_samples,)
        A single (categorical) feature column.

    y_col : ndarray of shape (n_samples,)
        Binary indicator column for a single label.

    Returns
    -------
    proba : ndarray of shape (n_samples,)
        `proba[i]` is the fraction of positive samples among all samples
        sharing the feature value `x_col[i]`.
    """
    _, inverse = np.unique(x_col, return_inverse=True)
    counts = np.bincount(inverse)
    positives = np.bincount(inverse, weights=y_col)
    return (positives / counts)[inverse]


@Substitution(
    random_state=_random_state_docstring,
)
class MLSMOTE(BaseOverSampler):
    """Oversample multilabel data using MLSMOTE.

    MLSMOTE (Multilabel Synthetic Minority Over-sampling Technique) generates
    synthetic instances for labels suffering from imbalance. Minority labels
    are detected with the imbalance ratio per label (`IRLbl`) and its mean
    (`MeanIR`). For each minority label, the nearest neighbors of every
    associated instance (within the set of instances carrying that label) are
    used to interpolate continuous features and vote nominal features and the
    synthetic labelset, following the SMOTE principle extended to multilabel
    data.

    Read more in the :ref:`User Guide <mlsmote>`.

    Parameters
    ----------
    sampling_strategy : str, default='auto'
        Labels targeted by the resampling. Only ``'auto'`` is supported: every
        label with an imbalance ratio (`IRLbl`) greater than the mean
        imbalance ratio (`MeanIR`) is oversampled, as described in the
        original paper. One synthetic sample is generated per instance
        carrying a minority label.

        .. versionadded:: 0.15

    {random_state}

    k_neighbors : int, default=5
        Number of nearest neighbors (within the instances carrying the
        minority label) used to construct synthetic samples. If fewer
        neighbors are available, all of them are used.

    categorical_features : array-like of int or bool, default=None
        Indices of the categorical features. Can either be an array of indices
        or a boolean mask of shape (n_features,). If ``None``, all features
        are treated as continuous. Categorical features contribute to the
        neighborhood search through the Value Difference Metric and their
        synthetic values are the most frequent value among the neighbors,
        while continuous features are linearly interpolated.

        .. versionadded:: 0.15

    labelset_strategy : str, default='ranking'
        Strategy used to assign a labelset to a synthetic sample from the
        labelsets of its seed instance and neighbors:

        - ``'ranking'``: keep the labels present in more than half of the
          seed instance and its neighbors;
        - ``'union'``: keep the labels present in at least one of them;
        - ``'intersection'``: keep the labels present in all of them.

        .. versionadded:: 0.15

    Attributes
    ----------
    sampling_strategy_ : dict
        Dictionary containing the information to sample the dataset. The keys
        are the label indices and the values are the number of synthetic
        samples generated for each label.

    n_features_in_ : int
        Number of features in the input dataset.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during `fit`. Defined only when `X` has feature
        names that are all strings.

    See Also
    --------
    SMOTE : Over-sample using SMOTE.

    SMOTEN : Over-sample using the SMOTE variant specifically for categorical
        features only.

    RandomOverSampler : Over-sample by randomly duplicating samples.

    Notes
    -----
    The implementation is based on [1]_. The neighborhood search combines the
    Euclidean distance over continuous features with the Value Difference
    Metric over nominal features, where the latter is computed with respect
    to the minority label being oversampled. Synthetic samples are computed
    from the original data and appended at the end of the resampling, which
    matches the reference implementation by the authors of [1]_.

    Unlike the other samplers in imbalanced-learn, `MLSMOTE` exclusively
    supports multilabel-indicator targets of shape
    `(n_samples, n_labels)` and does not support single-label targets.

    Supports multilabel resampling.

    References
    ----------
    .. [1] Charte, F. & Rivera Rivas, Antonio & Del Jesus, Maria Jose &
       Herrera, Francisco. (2015). "MLSMOTE: Approaching imbalanced multilabel
       learning through synthetic instance generation." Knowledge-Based
       Systems. 89. 385-397. 10.1016/j.knosys.2015.07.019.

    Examples
    --------
    >>> from sklearn.datasets import make_multilabel_classification
    >>> from imblearn.over_sampling import MLSMOTE
    >>> X, y = make_multilabel_classification(
    ...     n_samples=50, n_classes=3, n_labels=1, random_state=0)
    >>> print(f"Original samples: {{X.shape[0]}}")
    Original samples: 50
    >>> print(f"Label counts: {{y.sum(axis=0).tolist()}}")
    Label counts: [15, 21, 17]
    >>> sampler = MLSMOTE(random_state=0)
    >>> X_res, y_res = sampler.fit_resample(X, y)
    >>> print(f"Resampled samples: {{X_res.shape[0]}}")
    Resampled samples: 82
    >>> print(f"Label counts: {{y_res.sum(axis=0).tolist()}}")
    Label counts: [41, 27, 42]
    """

    _parameter_constraints: dict = {
        "sampling_strategy": [StrOptions({"auto"})],
        "random_state": ["random_state"],
        "k_neighbors": [Interval(numbers.Integral, 1, None, closed="left")],
        "categorical_features": ["array-like", None],
        "labelset_strategy": [StrOptions({"ranking", "union", "intersection"})],
    }

    def __init__(
        self,
        *,
        sampling_strategy="auto",
        random_state=None,
        k_neighbors=5,
        categorical_features=None,
        labelset_strategy="ranking",
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.categorical_features = categorical_features
        self.labelset_strategy = labelset_strategy

    def _check_X_y(self, X, y):
        """Validate X and accept multilabel-indicator targets only."""
        if sparse.issparse(X):
            raise ValueError(
                "MLSMOTE does not support sparse input X. Convert X to a dense"
                " array or dataframe."
            )
        y = check_array(
            y,
            accept_sparse=False,
            dtype=None,
            ensure_2d=False,
            ensure_all_finite=True,
        )
        if type_of_target(y) != "multilabel-indicator":
            raise ValueError(
                "MLSMOTE exclusively supports multilabel-indicator targets of"
                " shape (n_samples, n_labels). Got target of type"
                f" '{type_of_target(y)}' instead."
            )
        if not np.isin(y, [0, 1]).all():
            raise ValueError(
                "MLSMOTE requires a binary multilabel-indicator target with"
                " values in {0, 1}."
            )
        y = np.asarray(y, dtype=np.int64)
        if (y.sum(axis=0) == 0).any():
            raise ValueError(
                "MLSMOTE requires every label to contain at least one positive sample."
            )
        X = validate_data(
            self,
            X=X,
            y="no_validation",
            reset=True,
            dtype=None,
            accept_sparse=False,
            ensure_2d=True,
        )
        check_consistent_length(X, y)
        # `binarize_y` is always False: the multilabel target is used as-is.
        return X, y, False

    def _resolve_feature_types(self, n_features):
        """Split feature indices into categorical and continuous features."""
        if self.categorical_features is None:
            categorical = np.empty(0, dtype=np.int64)
        else:
            categorical = np.asarray(self.categorical_features)
            if categorical.dtype == bool:
                if categorical.shape != (n_features,):
                    raise ValueError(
                        "The boolean mask `categorical_features` should be of"
                        f" shape ({n_features},). Got {categorical.shape}."
                    )
                categorical = np.flatnonzero(categorical)
            else:
                categorical = np.unique(categorical.astype(np.int64))
                if (
                    categorical.size == 0
                    or categorical.min() < 0
                    or categorical.max() >= n_features
                ):
                    raise ValueError(
                        "Categorical feature indices should be between 0 and"
                        f" {n_features - 1}."
                    )
        continuous = np.setdiff1d(np.arange(n_features), categorical)
        return categorical, continuous

    @_fit_context(prefer_skip_nested_validation=True)
    def fit(self, X, y, **params):
        """Check inputs and statistics of the sampler.

        You should use ``fit_resample`` in all cases.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            Data array.

        y : array-like of shape (n_samples, n_labels)
            Multilabel-indicator target array.

        **params : dict
            Extra parameters to use by the sampler.

        Returns
        -------
        self : object
            Return the instance itself.
        """
        check_classification_targets(y)
        X, y, _ = self._check_X_y(X, y)
        minority_bags = self._minority_bags(y)
        self.sampling_strategy_ = {
            label: int(len(minority_bags.get(label, []))) for label in range(y.shape[1])
        }
        return self

    @_fit_context(prefer_skip_nested_validation=True)
    def fit_resample(self, X, y, **params):
        """Resample the dataset.

        Parameters
        ----------
        X : {array-like, dataframe} of shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like of shape (n_samples, n_labels)
            Multilabel-indicator target for each sample in X.

        **params : dict
            Extra parameters to use by the sampler.

        Returns
        -------
        X_resampled : {array-like, dataframe} of shape \
                (n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : array-like of shape (n_samples_new, n_labels)
            The corresponding multilabel-indicator target of `X_resampled`.
        """
        check_classification_targets(y)
        arrays_transformer = ArraysTransformer(X, y)
        X, y, _ = self._check_X_y(X, y)
        X_resampled, y_resampled = self._fit_resample(X, y)
        X_resampled, y_resampled = arrays_transformer.transform(
            X_resampled, y_resampled
        )
        return X_resampled, y_resampled

    def _minority_bags(self, y):
        """Map each minority label to the indices of its samples.

        A label is a minority label when its imbalance ratio (`IRLbl`) is
        greater than the mean imbalance ratio (`MeanIR`), as defined in the
        MLSMOTE paper.
        """
        counts = y.sum(axis=0).astype(np.float64)
        irlbl = counts.max() / counts
        mean_ir = irlbl.mean()
        return {
            label: np.flatnonzero(y[:, label])
            for label in range(y.shape[1])
            if irlbl[label] > mean_ir
        }

    def _fit_resample(self, X, y):
        random_state = check_random_state(self.random_state)
        n_samples, n_features = X.shape
        n_labels = y.shape[1]
        categorical, continuous = self._resolve_feature_types(n_features)

        try:
            X_continuous = np.asarray(X[:, continuous], dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "MLSMOTE requires continuous features to be numeric. Declare"
                " non-numeric features through `categorical_features`."
            ) from exc
        X_categorical = X[:, categorical]

        minority_bags = self._minority_bags(y)
        # Precompute P(y_label=1 | feature=value) on the original data for
        # every minority label and categorical feature. The Value Difference
        # Metric between two samples is derived from these probabilities.
        proba_per_label = {
            label: [
                _positive_proba_per_value(X_categorical[:, idx], y[:, label])
                for idx in range(X_categorical.shape[1])
            ]
            for label in minority_bags
        }

        X_new, y_new = [], []
        sampling_strategy_ = dict.fromkeys(range(n_labels), 0)
        for label, bag in minority_bags.items():
            bag_size = bag.shape[0]
            if bag_size <= 1:
                # A single sample has no neighbor to interpolate with.
                continue
            distances = self._pairwise_distances(
                X_continuous, X_categorical, proba_per_label[label], bag
            )
            order = np.argsort(distances, axis=1, kind="stable")
            n_neighbors = min(self.k_neighbors, bag_size - 1)
            neighbors = order[:, :n_neighbors]
            for seed_rank, seed in enumerate(bag):
                seed_neighbors = bag[neighbors[seed_rank]]
                reference = seed_neighbors[random_state.choice(n_neighbors)]
                X_new.append(
                    self._create_sample(
                        X,
                        X_continuous,
                        seed,
                        reference,
                        seed_neighbors,
                        categorical,
                        continuous,
                        random_state,
                    )
                )
                y_new.append(
                    self._create_labelset(y, seed, seed_neighbors, n_neighbors)
                )
            sampling_strategy_[label] = bag_size

        self.sampling_strategy_ = {
            label: int(count) for label, count in sampling_strategy_.items()
        }
        if not X_new:
            return X.copy(), y.copy()
        X_resampled = self._stack_features(X, np.array(X_new), categorical)
        y_resampled = np.vstack([y, np.array(y_new, dtype=y.dtype)])
        return X_resampled, y_resampled

    def _pairwise_distances(self, X_continuous, X_categorical, proba_per_feature, bag):
        """Compute MLSMOTE distances between the samples of `bag`.

        The distance combines the squared Euclidean distance over continuous
        features with the Value Difference Metric over categorical features,
        as defined in the MLSMOTE paper. The distance of a sample to itself
        is set to infinity so that it is never selected as its own neighbor.
        """
        bag_continuous = X_continuous[bag]
        squared_euclidean = (
            (bag_continuous[:, None, :] - bag_continuous[None, :, :]) ** 2
        ).sum(axis=-1)
        vdm = np.zeros_like(squared_euclidean)
        for feature_idx in range(X_categorical.shape[1]):
            proba = proba_per_feature[feature_idx][bag]
            # |P(y=1|a) - P(y=1|b)| + |P(y=0|a) - P(y=0|b)|
            # is 2 * |P(y=1|a) - P(y=1|b)| for binary labels.
            vdm += 2.0 * np.abs(proba[:, None] - proba[None, :])
        distances = np.sqrt(squared_euclidean + vdm)
        np.fill_diagonal(distances, np.inf)
        return distances

    def _create_sample(
        self,
        X,
        X_continuous,
        seed,
        reference,
        seed_neighbors,
        categorical,
        continuous,
        random_state,
    ):
        """Create the feature vector of a synthetic sample."""
        synthetic = np.empty(X.shape[1], dtype=object)
        steps = random_state.uniform(size=len(continuous))
        synthetic[continuous] = X_continuous[seed] + steps * (
            X_continuous[reference] - X_continuous[seed]
        )
        for position, feature in enumerate(categorical):
            synthetic[feature] = _most_frequent(X[seed_neighbors, feature])
        return synthetic

    def _create_labelset(self, y, seed, seed_neighbors, n_neighbors):
        """Create the labelset of a synthetic sample."""
        counts = y[seed] + y[seed_neighbors].sum(axis=0)
        if self.labelset_strategy == "ranking":
            selected = counts > (n_neighbors + 1) / 2
        elif self.labelset_strategy == "union":
            selected = counts >= 1
        else:  # intersection
            selected = counts == n_neighbors + 1
        return selected.astype(y.dtype)

    def _stack_features(self, X, X_new, categorical):
        """Stack synthetic samples below `X` preserving the input dtype."""
        if categorical.size == 0:
            X_new = np.asarray(X_new, dtype=np.float64)
        if X.dtype == object:
            return np.vstack([X, X_new])
        return np.vstack([X, X_new]).astype(X.dtype, copy=False)

    def _more_tags(self):
        return {
            "X_types": ["2darray", "dataframe"],
        }

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = False
        tags.input_tags.dataframe = True
        return tags
