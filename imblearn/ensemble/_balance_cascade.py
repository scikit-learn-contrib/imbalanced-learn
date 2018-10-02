"""Class to perform under-sampling using balace cascade."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

from collections import Counter

import numpy as np

from sklearn.base import ClassifierMixin, clone
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_random_state, safe_indexing
from sklearn.model_selection import cross_val_predict
from sklearn.utils.deprecation import deprecated

from .base import BaseEnsembleSampler
from ..under_sampling.base import BaseUnderSampler
from ..utils import check_sampling_strategy
from ..utils import Substitution
from ..utils._docstring import _random_state_docstring


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    random_state=_random_state_docstring)
@deprecated('BalanceCascade is deprecated in 0.4 and will be removed in 0.6.')
class BalanceCascade(BaseEnsembleSampler):
    """Create an ensemble of balanced sets by iteratively under-sampling the
    imbalanced dataset using an estimator.

    This method iteratively select subset and make an ensemble of the
    different sets. The selection is performed using a specific classifier.

    Parameters
    ----------
    {sampling_strategy}

    return_indices : bool, optional (default=True)
        Whether or not to return the indices of the samples randomly
        selected from the majority class.

    {random_state}

    n_max_subset : int or None, optional (default=None)
        Maximum number of subsets to generate. By default, all data from
        the training will be selected that could lead to a large number of
        subsets. We can probably deduce this number empirically.

    estimator : object, optional (default=KNeighborsClassifier())
        An estimator inherited from :class:`sklearn.base.ClassifierMixin` and
        having an attribute :func:`predict_proba`.

    bootstrap : bool, optional (default=True)
        Whether to bootstrap the data before each iteration.

    ratio : str, dict, or callable
        .. deprecated:: 0.4
           Use the parameter ``sampling_strategy`` instead. It will be removed
           in 0.6.

    Notes
    -----
    The method is described in [1]_.

    Supports multi-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    See also
    --------
    BalancedBaggingClassifier, EasyEnsemble

    References
    ----------
    .. [1] X. Y. Liu, J. Wu and Z. H. Zhou, "Exploratory Undersampling for
       Class-Imbalance Learning," in IEEE Transactions on Systems, Man, and
       Cybernetics, Part B (Cybernetics), vol. 39, no. 2, pp. 539-550,
       April 2009.

    Examples
    --------

    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from imblearn.ensemble import \
BalanceCascade # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> bc = BalanceCascade(random_state=42)
    >>> X_res, y_res = bc.fit_resample(X, y)
    >>> print('Resampled dataset shape %s' % Counter(y_res[0])) \
    # doctest: +ELLIPSIS
    Resampled dataset shape Counter({{...}})

    """

    def __init__(self,
                 sampling_strategy='auto',
                 return_indices=False,
                 random_state=None,
                 n_max_subset=None,
                 estimator=None,
                 ratio=None):
        super(BalanceCascade, self).__init__(
            sampling_strategy=sampling_strategy, ratio=ratio)
        self.random_state = random_state
        self.return_indices = return_indices
        self.estimator = estimator
        self.n_max_subset = n_max_subset

    def _validate_estimator(self):
        """Private function to create the classifier"""

        if (self.estimator is not None and
                isinstance(self.estimator, ClassifierMixin) and
                hasattr(self.estimator, 'predict')):
            self.estimator_ = clone(self.estimator)
        elif self.estimator is None:
            self.estimator_ = KNeighborsClassifier()
        else:
            raise ValueError('Invalid parameter `estimator`. Got {}.'.format(
                type(self.estimator)))

    def _fit_resample(self, X, y):
        self._validate_estimator()

        self.sampling_strategy_ = check_sampling_strategy(
            self.sampling_strategy, y, 'under-sampling')

        random_state = check_random_state(self.random_state)

        # array to know which samples are available to be taken
        samples_mask = np.ones(y.shape, dtype=bool)

        # where the different set will be stored
        idx_under = []

        n_subsets = 0
        b_subset_search = True
        while b_subset_search:
            target_stats = Counter(
                safe_indexing(y, np.flatnonzero(samples_mask)))
            # store the index of the data to under-sample
            index_under_sample = np.empty((0, ), dtype=np.int)
            # value which will be picked at each round
            index_constant = np.empty((0, ), dtype=np.int)
            for target_class in target_stats.keys():
                if target_class in self.sampling_strategy_.keys():
                    n_samples = self.sampling_strategy_[target_class]
                    # extract the data of interest for this round from the
                    # current class
                    index_class = np.flatnonzero(y == target_class)
                    index_class_interest = index_class[samples_mask[
                        y == target_class]]
                    y_class = safe_indexing(y, index_class_interest)
                    # select randomly the desired features
                    index_target_class = random_state.choice(
                        range(y_class.size), size=n_samples, replace=False)
                    index_under_sample = np.concatenate(
                        (index_under_sample,
                         index_class_interest[index_target_class]),
                        axis=0)
                else:
                    index_constant = np.concatenate(
                        (index_constant, np.flatnonzero(y == target_class)),
                        axis=0)

            # store the set created
            n_subsets += 1
            subset_indices = np.concatenate(
                (index_under_sample, index_constant), axis=0)
            idx_under.append(subset_indices)

            # fit and predict using cross validation
            X_subset = safe_indexing(X, subset_indices)
            y_subset = safe_indexing(y, subset_indices)
            pred = cross_val_predict(self.estimator_, X_subset, y_subset, cv=3)
            # extract the prediction about the targeted classes only
            pred_target = pred[:index_under_sample.size]
            index_classified = index_under_sample[pred_target == safe_indexing(
                y_subset, range(index_under_sample.size))]
            samples_mask[index_classified] = False

            # check the stopping criterion
            if self.n_max_subset is not None:
                if n_subsets == self.n_max_subset:
                    b_subset_search = False
            # check that there is enough samples for another round
            target_stats = Counter(
                safe_indexing(y, np.flatnonzero(samples_mask)))
            for target_class in self.sampling_strategy_.keys():
                if (target_stats[target_class] <
                        self.sampling_strategy_[target_class]):
                    b_subset_search = False

        X_resampled, y_resampled = [], []
        for indices in idx_under:
            X_resampled.append(safe_indexing(X, indices))
            y_resampled.append(safe_indexing(y, indices))

        if self.return_indices:
            return (np.array(X_resampled), np.array(y_resampled),
                    np.array(idx_under))
        else:
            return np.array(X_resampled), np.array(y_resampled)
