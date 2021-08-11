"""Class to perform under-sampling by removing Tomek's links."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Fernando Nogueira
#          Christos Aridas
# License: MIT

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import _safe_indexing

from ..base import BaseCleaningSampler
from ...utils import Substitution
from ...utils._docstring import _n_jobs_docstring
from ...utils._validation import _deprecate_positional_args


@Substitution(
    sampling_strategy=BaseCleaningSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
)
class TomekLinks(BaseCleaningSampler):
    """Under-sampling by removing Tomek's links.

    Read more in the :ref:`User Guide <tomek_links>`.

    Parameters
    ----------
    {sampling_strategy}

    {n_jobs}

    Attributes
    ----------
    sample_indices_ : ndarray of shape (n_new_samples,)
        Indices of the samples selected.

        .. versionadded:: 0.4

    See Also
    --------
    EditedNearestNeighbours : Undersample by samples edition.

    CondensedNearestNeighbour : Undersample by samples condensation.

    RandomUnderSampling : Randomly under-sample the dataset.

    Notes
    -----
    This method is based on [1]_.

    Supports multi-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    References
    ----------
    .. [1] I. Tomek, "Two modifications of CNN", in Systems, Man, and
       Cybernetics, IEEE Transactions on, vol. 6, pp 769-772, 1976.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.under_sampling import \
TomekLinks # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> tl = TomekLinks()
    >>> X_res, y_res = tl.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res))
    Resampled dataset shape Counter({{1: 897, 0: 100}})
    """

    @_deprecate_positional_args
    def __init__(self, *, sampling_strategy="auto", n_jobs=None):
        super().__init__(sampling_strategy=sampling_strategy)
        self.n_jobs = n_jobs

    @staticmethod
    def is_tomek(y, nn_index, class_type):
        """Detect if samples are Tomek's link.

        More precisely, it uses the target vector and the first neighbour of
        every sample point and looks for Tomek pairs. Returning a boolean
        vector with True for majority Tomek links.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target vector of the data set, necessary to keep track of whether a
            sample belongs to minority class.

        nn_index : ndarray of shape (len(y),)
            Index with the closest nearest neighbour to a sample.

        class_type : int or str
            The label of the minority class.

        Returns
        -------
        is_tomek : ndarray of shape (len(y), )
            Boolean vector of len( # samples ), with True for majority samples
            that are Tomek links.
        """
        links = np.zeros(len(y), dtype=bool)

        # find which class not to consider
        class_excluded = [c for c in np.unique(y) if c not in class_type]

        # there is a Tomek link between two samples if they are nearest
        # neighbors of each other, and from a different class.
        for index_sample, target_sample in enumerate(y):
            if target_sample in class_excluded:
                continue

            if y[nn_index[index_sample]] != target_sample:
                # corroborate that they are neighbours of each other:
                # (if A's closest neighbour is B, but B's closest neighbour
                # is C, then A and B are not a Tomek link)
                if nn_index[nn_index[index_sample]] == index_sample:
                    links[index_sample] = True

        return links

    def _fit_resample(self, X, y):
        # Find the nearest neighbour of every point
        nn = NearestNeighbors(n_neighbors=2, n_jobs=self.n_jobs)
        nn.fit(X)
        nns = nn.kneighbors(X, return_distance=False)[:, 1]

        links = self.is_tomek(y, nns, self.sampling_strategy_)
        self.sample_indices_ = np.flatnonzero(np.logical_not(links))

        return (
            _safe_indexing(X, self.sample_indices_),
            _safe_indexing(y, self.sample_indices_),
        )

    def _more_tags(self):
        return {"sample_indices": True}
