"""Class to perform cleaning and selective pre-processing using SPIDER"""

# Authors: Matthew Eding
# License: MIT


from numbers import Integral

import numpy as np
from scipy import sparse
from scipy import stats

from sklearn.utils import safe_mask
from sklearn.utils import _safe_indexing

from .base import BasePreprocessSampler
from ...utils import check_neighbors_object
from ...utils import Substitution
from ...utils._docstring import _n_jobs_docstring

SEL_KIND = ("weak", "relabel", "strong")


@Substitution(
    sampling_strategy=BasePreprocessSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
)
class SPIDER(BasePreprocessSampler):
    """Perform filtering and over-sampling using Selective Pre-processing of
    Imbalanced Data (SPIDER) sampling approach for imbalanced datasets.

    Read more in the :ref:`User Guide <combine>`.

    Parameters
    ----------
    {sampling_strategy}

    kind : str (default='weak')
        Possible choices are:

            ``'weak'``: Amplify noisy minority class samples based on the
            number of safe majority neighbors.

            ``'relabel'``: Perform ``'weak'`` amplification and then relabel
            noisy majority neighbors for each noisy minority class sample.

            ``'strong'``: Amplify all minority class samples by an extra
            ``additional_neighbors`` if the sample is classified incorrectly
            by its neighbors. Otherwise each minority sample is amplified in a
            manner akin to ``'weak'`` amplification.

    n_neighbors : int or object, optional (default=3)
        If ``int``, number of nearest neighbours to used to construct synthetic
        samples.  If object, an estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.

    additional_neighbors : int, optional (default=2)
        The number to add to amplified samples during if ``kind`` is
        ``'strong'``. This has no effect otherwise.

    {n_jobs}

    Notes
    -----
    The implementation is based on [1]_ and [2]_.

    Supports multi-class resampling. A one-vs.-rest scheme is used.

    See also
    --------
    NeighborhoodClearingRule and RandomOverSampler

    References
    ----------
    .. [1] Stefanowski, J., & Wilk, S, "Selective pre-processing of imbalanced
       data for improving classification performance," In: Song, I.-Y., Eder,
       J., Nguyen, T.M. (Eds.): DaWaK 2008, LNCS, vol. 5182, pp. 283–292.
       Springer, Heidelberg, 2008.

    .. [2] Błaszczyński, J., Deckert, M., Stefanowski, J., & Wilk, S,
       "Integrating Selective Pre-processing of Imbalanced Data with Ivotes
       Ensemble," In: M. Szczuka et al. (Eds.): RSCTC 2010, LNAI 6086, pp.
       148–157, 2010.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.combine import \
SPIDER # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000,
    ... random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> spider = SPIDER()
    >>> X_res, y_res = spider.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{1: 897, 0: 115}})
    """

    def __init__(
        self,
        sampling_strategy="auto",
        kind="weak",
        n_neighbors=3,
        additional_neighbors=2,
        n_jobs=None,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.kind = kind
        self.n_neighbors = n_neighbors
        self.additional_neighbors = additional_neighbors
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Create the necessary objects for SPIDER"""
        self.nn_ = check_neighbors_object(
            "n_neighbors", self.n_neighbors, additional_neighbor=1)
        self.nn_.set_params(**{"n_jobs": self.n_jobs})

        if self.kind not in SEL_KIND:
            raise ValueError(
                'The possible "kind" of algorithm are "weak", "relabel",'
                ' and "strong". Got {} instead.'.format(self.kind)
            )

        if self.additional_neighbors < 1:
            raise ValueError("additional_neighbors must be at least 1.")

        if not isinstance(self.additional_neighbors, Integral):
            raise TypeError("additional_neighbors must be an integer.")

    def _locate_neighbors(self, X, additional=False):
        """Find nearest neighbors for samples.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The feature samples to find neighbors for.

        additional : bool, optional (default=False)
            Flag to indicate whether to increase ``n_neighbors`` by
            ``additional_neighbors``.

        Returns
        -------
        nn_indices : ndarray, shape (n_samples, n_neighbors)
            Indices of the nearest neighbors for the subset.
        """
        n_neighbors = self.nn_.n_neighbors
        if additional:
            n_neighbors += self.additional_neighbors

        nn_indices = self.nn_.kneighbors(
            X, n_neighbors, return_distance=False)[:, 1:]
        return nn_indices

    def _knn_correct(self, X, y, additional=False):
        """Apply KNN to classify samples.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The feature samples to classify.

        y : ndarray, shape (n_samples,)
            The label samples to classify.

        additional : bool, optional (default=False)
            Flag to indicate whether to increase ``n_neighbors`` by
            additional_neighbors``.

        Returns
        -------
        is_correct : ndarray[bool], shape (n_samples,)
            Mask that indicates if KNN classifed samples correctly.
        """
        try:
            nn_indices = self._locate_neighbors(X, additional)
        except ValueError:
            return np.empty(0, dtype=bool)

        mode, _ = stats.mode(self._y[nn_indices], axis=1)
        is_correct = (y == mode.ravel())
        return is_correct

    def _amplify(self, X, y, additional=False):
        """In-place amplification of samples based on their neighborhood
        counts of samples that are safe and belong to the other class(es).
        Returns ``nn_indices`` for relabel usage.

        Parameters
        ----------
        X : ndarray, shape (n_samples, n_features)
            The feature samples to amplify.

        y : ndarray, shape (n_samples,)
            The label samples to amplify.

        additional : bool, optional (default=False)
            Flag to indicate whether to amplify with ``additional_neighbors``.

        Returns
        -------
        nn_indices : ndarray, shape (n_samples, n_neighbors)
            Indices of the nearest neighbors for the subset.
        """
        try:
            nn_indices = self._locate_neighbors(X, additional)
        except ValueError:
            return np.empty(0, dtype=int)

        amplify_amounts = np.isin(
            nn_indices, self._amplify_indices).sum(axis=1)

        X_parts = []
        y_parts = []
        for amount in filter(bool, np.unique(amplify_amounts)):
            mask = safe_mask(X, amplify_amounts == amount)
            X_part = X[mask]
            y_part = y[mask]
            X_parts.extend([X_part] * amount)
            y_parts.extend([y_part] * amount)

            if sparse.issparse(X):
                X_new = sparse.vstack(X_parts)
            else:
                X_new = np.vstack(X_parts)
            y_new = np.hstack(y_parts)

        self._X_resampled.append(X_new)
        self._y_resampled.append(y_new)
        return nn_indices

    def _fit_resample(self, X, y):
        self._validate_estimator()

        self._X_resampled = []
        self._y_resampled = []
        self._y = y.copy()

        self.nn_.fit(X)
        is_safe = self._knn_correct(X, y)

        strategy = self.sampling_strategy_
        for class_sample in filter(strategy.get, strategy):
            is_class = (y == class_sample)
            self._amplify_indices = np.flatnonzero(~is_class & is_safe)
            discard_indices = np.flatnonzero(~is_class & ~is_safe)

            class_noisy_indices = np.flatnonzero(is_class & ~is_safe)
            X_class_noisy = _safe_indexing(X, class_noisy_indices)
            y_class_noisy = y[class_noisy_indices]

            if self.kind in ("weak", "relabel"):
                nn_indices = self._amplify(X_class_noisy, y_class_noisy)

                if self.kind == "relabel":
                    relabel_mask = np.isin(nn_indices, discard_indices)
                    relabel_indices = np.unique(nn_indices[relabel_mask])
                    self._y[relabel_indices] = class_sample
                    discard_indices = np.setdiff1d(
                        discard_indices, relabel_indices)

            elif self.kind == "strong":
                class_safe_indices = np.flatnonzero(is_class & is_safe)
                X_class_safe = _safe_indexing(X, class_safe_indices)
                y_class_safe = y[class_safe_indices]
                self._amplify(X_class_safe, y_class_safe)

                is_correct = self._knn_correct(
                    X_class_noisy, y_class_noisy, additional=True)

                X_correct = X_class_noisy[
                    safe_mask(X_class_noisy, is_correct)]
                y_correct = y_class_noisy[is_correct]
                self._amplify(X_correct, y_correct)

                X_incorrect = X_class_noisy[
                    safe_mask(X_class_noisy, ~is_correct)]
                y_incorrect = y_class_noisy[~is_correct]
                self._amplify(X_incorrect, y_incorrect, additional=True)
            else:
                raise NotImplementedError(self.kind)

        discard_mask = np.ones_like(y, dtype=bool)
        try:
            discard_mask[discard_indices] = False
        except UnboundLocalError:
            pass

        X_resampled = self._X_resampled
        y_resampled = self._y_resampled

        X_resampled.append(X[safe_mask(X, discard_mask)])
        y_resampled.append(self._y[discard_mask])

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:
            X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        del self._X_resampled, self._y_resampled, self._y
        self._amplify_indices = None
        return X_resampled, y_resampled
