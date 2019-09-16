"""Class to perform cleaning and selective pre-processing using SPIDER"""

# Author: Matthew Eding
# License: MIT


import numpy as np
from scipy import sparse
from scipy import stats

from sklearn.utils import safe_indexing, safe_mask

from ..over_sampling.base import BaseOverSampler
from ..under_sampling.base import BaseCleaningSampler
from ..utils import check_neighbors_object
from ..utils import Substitution


@Substitution(sampling_strategy=BaseCleaningSampler._sampling_strategy_docstring)
class SPIDER(BaseCleaningSampler, BaseOverSampler):
    """Perform filtering and over-sampling using Selective Pre-processing of
    Imbalanced Data (SPIDER) sampling approach for imbalanced datasets.

    TODO Read more in the :ref:`User Guide <spider>`.

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

    n_neighbors : int or object, optional (default=5)
        If ``int``, number of nearest neighbours to used to construct synthetic
        samples.  If object, an estimator that inherits from
        :class:`sklearn.neighbors.base.KNeighborsMixin` that will be used to
        find the k_neighbors.

    additional_neighbors : int, optional (default=2)
        The number to add to amplified samples during if ``kind`` is
        ``'strong'``. This has no effect otherwise.

    n_jobs : int, optional (default=1)
        Number of threads to run the algorithm when it is possible.

    Attributes
    ----------
    discarded_ : TODO
        TODO

    relabeled_ : TODO
        TODO

    Notes
    -----
    The implementation is based on [1]_, [2]_ and [3]_.

    TODO Supports multi-class resampling. A one-vs.-rest scheme is used.

    See also
    --------
    SMOTE : Over-sample using SMOTE.

    References
    ----------
    .. [1] Stefanowski, J., & Wilk, S, "Improving rule based classifiers
       induced by MODLEM by selective pre-processing of imbalanced data," In:
       Proc. of the RSKD Workshop at ECML/PKDD, pp. 54–65, 2007.

    .. [2] Stefanowski, J., & Wilk, S, "Selective pre-processing of imbalanced
       data for improving classification performance," In: Song, I.-Y., Eder,
       J., Nguyen, T.M. (Eds.): DaWaK 2008, LNCS, vol. 5182, pp. 283–292.
       Springer, Heidelberg, 2008.

    .. [3] Błaszczyński, J., Deckert, M., Stefanowski, J., & Wilk, S,
       "Integrating Selective Pre-processing of Imbalanced Data with Ivotes
       Ensemble," In: M. Szczuka et al. (Eds.): RSCTC 2010, LNAI 6086, pp.
       148–157, 2010.

    Examples
    --------
    TODO
    """

    def __init__(
        self,
        sampling_strategy='auto',
        kind='weak',
        n_neighbors=3,
        additional_neighbors=2,
        n_jobs=1,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.kind = kind
        self.n_neighbors = n_neighbors
        self.additional_neighbors = min(1, int(additional_neighbors))
        self.n_jobs = n_jobs

    def _validate_estimator(self):
        """Create the necessary objects for SPIDER"""
        self.nn_ = check_neighbors_object(
            'n_neighbors', self.n_neighbors, additional_neighbor=1)
        self.nn_.set_params(**{'n_jobs': self.n_jobs})

        if self.kind not in ('weak', 'relabel', 'strong'):
            raise ValueError('The possible "kind" of algorithm are '
                             '"weak", "relabel", and "strong".'
                             'Got {} instead.'.format(self.kind))

    def _locate_neighbors(self, X, additional=False):
        """Find nearest neighbors for samples.

        Parameters
        ----------
        X : ndarray, size(m_samples, n_features)
            The feature samples to find neighbors for.

        additional : bool, optional (defaul=False)
            Flag to indicate whether to increase ``n_neighbors`` by ``additional_neighbors``.

        Returns
        -------
        nn_indices : ndarray, size(TODO)
            Indices of the nearest neighbors for the subset.
        """
        n_neighbors = self.nn_.n_neighbors
        if additional:
            n_neighbors += self.additional_neighbors

        nn_indices = self.nn_.kneighbors(X, n_neighbors, return_distance=False)[:, 1:]
        return nn_indices

    def _knn_correct(self, X, y, additional=False):
        """Apply KNN to classify samples.

        Parameters
        ----------
        X : ndarray, size(m_samples, n_features)
            The feature samples to classify.

        y : ndarray, size(m_samples,)
            The label samples to classify.

        additional : bool, optional (defaul=False)
            Flag to indicate whether to increase ``n_neighbors`` by ``additional_neighbors``.

        Returns
        -------
        is_correct : ndarray[bool], size(m_samples,)
            Mask that indicates if KNN classifed samples correctly.
        """
        try:
            nn_indices = self._locate_neighbors(X, additional)
        except ValueError:
            return np.empty(0, dtype=bool) # TODO: check if this works
        mode, _ = stats.mode(self._y[nn_indices], axis=1)
        is_correct = (y == mode.ravel())
        return is_correct

    def _amplify(self, X, y, additional=False):
        """In-place amplification of samples based on their neighborhood
        counts of samples that are safe and belong to the other class(es).
        Returns ``nn_indices`` for relabel usage.

        Parameters
        ----------
        X : ndarray, size(m_samples, n_features)
            The feature samples to amplify.

        y : ndarray, size(m_samples,)
            The label samples to amplify.

        additional : bool, optional (defaul=False)
            Flag to indicate whether to amplify with ``additional_neighbors``.

        Returns
        -------
        nn_indices : TODO
            TODO
        """
        try:
            nn_indices = self._locate_neighbors(X, additional)
        except ValueError:
            return np.empty(0, dtype=int)
        
        amplify_amounts = np.isin(nn_indices, self._amplify_indices).sum(axis=1)

        if additional:
            amplify_amounts += self.additional_neighbors

        if sparse.issparse(X):
            X_parts = []
            for amount in filter(bool, np.unique(amplify_amounts)):
                X_part = X[safe_mask(X, amplify_amounts == amount)]
                X_parts.extend([X_part] * amount)
            X_new = sparse.vstack(X_parts)
        else:
            X_new = np.repeat(X, amplify_amounts, axis=0)
        
        y_new = np.repeat(y, amplify_amounts)
        self._X_resampled.append(X_new)
        self._y_resampled.append(y_new)
        return nn_indices

    def _fit_resample(self, X, y):
        self._validate_estimator()

        self._X_resampled = []
        self._y_resampled = []
        self._X = X # do I need this one for X?
        self._y = y

        self.nn_.fit(X)
        is_safe = self._knn_correct(X, y)

        strategy = self.sampling_strategy_
        #TODO: double check that class_sample means the value that indicates which class
        for class_sample in filter(strategy.get, strategy):
            is_class = (y == class_sample)
            self._amplify_indices = np.flatnonzero(~is_class & is_safe)
            #TODO see what some cleaning samplers call idxs that are to be removed
            discard_indices = np.flatnonzero(~is_class & ~is_safe)

            class_noisy_indices = np.flatnonzero(is_class & ~is_safe)
            X_class_noisy = safe_indexing(X, class_noisy_indices)
            y_class_noisy = safe_indexing(y, class_noisy_indices)

            self.relabeled_ = np.empty(0, dtype=int)

            if self.kind in ('weak', 'relabel'):
                nn_indices = self._amplify(X_class_noisy, y_class_noisy)

                if self.kind == 'relabel':
                    relabel_mask = np.isin(nn_indices, discard_indices)
                    relabel_indices = np.unique(nn_indices[relabel_mask])
                    y[relabel_indices] = class_sample
                    discard_indices = np.setdiff1d(discard_indices, relabel_indices)
                    self.relabeled_ = relabel_indices                    

            elif self.kind == 'strong':
                class_safe_indices = np.flatnonzero(is_class & is_safe)
                X_class_safe = safe_indexing(X, class_safe_indices)
                y_class_safe = safe_indexing(y, class_safe_indices)
                self._amplify(X_class_safe, y_class_safe)

                is_correct = self._knn_correct(X_class_noisy, y_class_noisy, additional=True)

                X_correct = X_class_noisy[is_correct]
                y_correct = y_class_noisy[is_correct]
                self._amplify(X_correct, y_correct)

                X_incorrect = X_class_noisy[~is_correct]
                y_incorrect = y_class_noisy[~is_correct]
                self._amplify(X_incorrect, y_incorrect, additional=True)
            else:
                raise NotImplementedError(self.kind)

        
        self.discarded_ = discard_indices
        discard_mask = np.ones_like(y, dtype=bool)
        discard_mask[discard_indices] = False

        X_resampled = self._X_resampled
        y_resampled = self._y_resampled

        X_resampled.append(X[safe_mask(X, discard_mask)])
        y_resampled.append(y[discard_mask])

        if sparse.issparse(X):
            X_resampled = sparse.vstack(X_resampled, format=X.format)
        else:
            X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)

        del self._X_resampled, self._y_resampled
        return X_resampled, y_resampled
