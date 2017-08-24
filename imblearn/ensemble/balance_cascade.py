"""Class to perform under-sampling using balace cascade."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import warnings

from collections import Counter

import numpy as np

from sklearn.base import ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import check_random_state, safe_indexing
from sklearn.externals.six import string_types
from sklearn.model_selection import cross_val_predict

from .base import BaseEnsembleSampler
from ..utils import check_ratio


class BalanceCascade(BaseEnsembleSampler):
    """Create an ensemble of balanced sets by iteratively under-sampling the
    imbalanced dataset using an estimator.

    This method iteratively select subset and make an ensemble of the
    different sets. The selection is performed using a specific classifier.

    Read more in the :ref:`User Guide <ensemble_samplers>`.

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

    return_indices : bool, optional (default=True)
        Whether or not to return the indices of the samples randomly
        selected from the majority class.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number
        generator; If ``RandomState`` instance, random_state is the random
        number generator; If ``None``, the random number generator is the
        ``RandomState`` instance used by ``np.random``.

    n_max_subset : int or None, optional (default=None)
        Maximum number of subsets to generate. By default, all data from
        the training will be selected that could lead to a large number of
        subsets. We can probably deduce this number empirically.

    classifier : str, optional (default=None)
        The classifier that will be selected to confront the prediction
        with the real labels. The choices are the following: ``'knn'``,
        ``'decision-tree'``, ``'random-forest'``, ``'adaboost'``,
        ``'gradient-boosting'``, and ``'linear-svm'``.

        .. deprecated:: 0.2
           ``classifier`` is deprecated from 0.2 and will be replaced in 0.4.
           Use ``estimator`` instead.

    estimator : object, optional (default=KNeighborsClassifier())
        An estimator inherited from :class:`sklearn.base.ClassifierMixin` and
        having an attribute :func:`predict_proba`.

    bootstrap : bool, optional (default=True)
        Whether to bootstrap the data before each iteration.

    **kwargs : keywords
        The parameters associated with the classifier provided.

        .. deprecated:: 0.2
           ``**kwargs`` has been deprecated from 0.2 and will be replaced in
           0.4. Use ``estimator`` object instead to pass parameters associated
           to an estimator.

    Notes
    -----
    The method is described in [1]_.

    Supports mutli-class resampling. A one-vs.-rest scheme is used as
    originally proposed in [1]_.

    See :ref:`sphx_glr_auto_examples_ensemble_plot_balance_cascade.py`.

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
    >>> print('Original dataset shape {}'.format(Counter(y)))
    Original dataset shape Counter({1: 900, 0: 100})
    >>> bc = BalanceCascade(random_state=42)
    >>> X_res, y_res = bc.fit_sample(X, y)
    >>> print('Resampled dataset shape {}'.format(Counter(y_res[0]))) \
    # doctest: +ELLIPSIS
    Resampled dataset shape Counter({...})

    """

    def __init__(self,
                 ratio='auto',
                 return_indices=False,
                 random_state=None,
                 n_max_subset=None,
                 classifier=None,
                 estimator=None,
                 **kwargs):
        super(BalanceCascade, self).__init__(ratio=ratio,
                                             random_state=random_state)
        self.return_indices = return_indices
        self.classifier = classifier
        self.estimator = estimator
        self.n_max_subset = n_max_subset
        self.kwargs = kwargs

    def fit(self, X, y):
        """Find the classes statistics before to perform sampling.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Matrix containing the data which have to be sampled.

        y : array-like, shape (n_samples,)
            Corresponding label for each sample in X.

        Returns
        -------
        self : object,
            Return self.

        """
        super(BalanceCascade, self).fit(X, y)
        self.ratio_ = check_ratio(self.ratio, y, 'under-sampling')
        return self

    def _validate_estimator(self):
        """Private function to create the classifier"""

        if self.classifier is not None:
            warnings.warn('`classifier` will be replaced in version'
                          ' 0.4. Use a `estimator` instead.',
                          DeprecationWarning)
            self.estimator = self.classifier

        if (self.estimator is not None and
                isinstance(self.estimator, ClassifierMixin) and
                hasattr(self.estimator, 'predict')):
            self.estimator_ = self.estimator
        elif self.estimator is None:
            self.estimator_ = KNeighborsClassifier()
        # To be removed in 0.4
        elif (self.estimator is not None and
              isinstance(self.estimator, string_types)):
            warnings.warn('`estimator` will be replaced in version'
                          ' 0.4. Use a classifier object instead of a string.',
                          DeprecationWarning)
            # Define the classifier to use
            if self.estimator == 'knn':
                self.estimator_ = KNeighborsClassifier(**self.kwargs)
            elif self.estimator == 'decision-tree':
                from sklearn.tree import DecisionTreeClassifier
                self.estimator_ = DecisionTreeClassifier(
                    random_state=self.random_state, **self.kwargs)
            elif self.estimator == 'random-forest':
                from sklearn.ensemble import RandomForestClassifier
                self.estimator_ = RandomForestClassifier(
                    random_state=self.random_state, **self.kwargs)
            elif self.estimator == 'adaboost':
                from sklearn.ensemble import AdaBoostClassifier
                self.estimator_ = AdaBoostClassifier(
                    random_state=self.random_state, **self.kwargs)
            elif self.estimator == 'gradient-boosting':
                from sklearn.ensemble import GradientBoostingClassifier
                self.estimator_ = GradientBoostingClassifier(
                    random_state=self.random_state, **self.kwargs)
            elif self.estimator == 'linear-svm':
                from sklearn.svm import LinearSVC
                self.estimator_ = LinearSVC(
                    random_state=self.random_state, **self.kwargs)
            else:
                raise NotImplementedError
        else:
            raise ValueError('Invalid parameter `estimator`. Got {}.'.format(
                type(self.estimator)))

        self.logger.debug(self.estimator_)

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
(n_subset, n_samples_new, n_features)
            The array containing the resampled data.

        y_resampled : ndarray, shape (n_subset, n_samples_new)
            The corresponding label of `X_resampled`

        idx_under : ndarray, shape (n_subset, n_samples, )
            If `return_indices` is `True`, a boolean array will be returned
            containing the which samples have been selected.

        """
        self._validate_estimator()

        random_state = check_random_state(self.random_state)

        # array to know which samples are available to be taken
        samples_mask = np.ones(y.shape, dtype=bool)

        # where the different set will be stored
        idx_under = []

        n_subsets = 0
        b_subset_search = True
        while b_subset_search:
            target_stats = Counter(safe_indexing(
                y, np.flatnonzero(samples_mask)))
            # store the index of the data to under-sample
            index_under_sample = np.empty((0, ), dtype=y.dtype)
            # value which will be picked at each round
            index_constant = np.empty((0, ), dtype=y.dtype)
            for target_class in target_stats.keys():
                if target_class in self.ratio_.keys():
                    n_samples = self.ratio_[target_class]
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
                        (index_constant,
                         np.flatnonzero(y == target_class)),
                        axis=0)

            # store the set created
            n_subsets += 1
            subset_indices = np.concatenate((index_under_sample,
                                             index_constant), axis=0)
            idx_under.append(subset_indices)

            # fit and predict using cross validation
            X_subset = safe_indexing(X, subset_indices)
            y_subset = safe_indexing(y, subset_indices)
            pred = cross_val_predict(self.estimator_, X_subset, y_subset)
            # extract the prediction about the targeted classes only
            pred_target = pred[:index_under_sample.size]
            index_classified = index_under_sample[
                pred_target == safe_indexing(y_subset,
                                             range(index_under_sample.size))]
            samples_mask[index_classified] = False

            # check the stopping criterion
            if self.n_max_subset is not None:
                if n_subsets == self.n_max_subset:
                    b_subset_search = False
            # check that there is enough samples for another round
            target_stats = Counter(safe_indexing(
                y, np.flatnonzero(samples_mask)))
            for target_class in self.ratio_.keys():
                if target_stats[target_class] < self.ratio_[target_class]:
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
