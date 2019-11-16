"""Class to perform under-sampling using easy ensemble."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numbers

import numpy as np

from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

from ..under_sampling import RandomUnderSampler
from ..under_sampling.base import BaseUnderSampler
from ..utils import Substitution, check_target_type
from ..utils._docstring import _n_jobs_docstring
from ..utils._docstring import _random_state_docstring
from ..pipeline import Pipeline

MAX_INT = np.iinfo(np.int32).max


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class EasyEnsembleClassifier(BaggingClassifier):
    """Bag of balanced boosted learners also known as EasyEnsemble.

    This algorithm is known as EasyEnsemble [1]_. The classifier is an
    ensemble of AdaBoost learners trained on different balanced boostrap
    samples. The balancing is achieved by random under-sampling.

    Read more in the :ref:`User Guide <boosting>`.

    Parameters
    ----------
    n_estimators : int, default=10
        Number of AdaBoost learners in the ensemble.

    base_estimator : object, default=AdaBoostClassifier()
        The base AdaBoost classifier used in the inner ensemble. Note that you
        can set the number of inner learner by passing your own instance.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble.

    {sampling_strategy}

    replacement : bool, default=False
        Whether or not to sample randomly with replacement or not.

    {n_jobs}

    {random_state}

    verbose : int, optional (default=0)
        Controls the verbosity of the building process.

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    estimators_ : list of estimators
        The collection of fitted base estimators.

    classes_ : array, shape (n_classes,)
        The classes labels.

    n_classes_ : int or list
        The number of classes.

    See Also
    --------
    BalancedBaggingClassifier : Bagging classifier for which each base
        estimator is trained on a balanced bootstrap.

    BalancedRandomForestClassifier : Random forest applying random-under
        sampling to balance the different bootstraps.

    RUSBoostClassifier : AdaBoost classifier were each bootstrap is balanced
        using random-under sampling at each round of boosting.

    Notes
    -----
    The method is described in [1]_.

    Supports multi-class resampling by sampling each class independently.

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
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import confusion_matrix
    >>> from imblearn.ensemble import \
EasyEnsembleClassifier # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     random_state=0)
    >>> eec = EasyEnsembleClassifier(random_state=42)
    >>> eec.fit(X_train, y_train) # doctest: +ELLIPSIS
    EasyEnsembleClassifier(...)
    >>> y_pred = eec.predict(X_test)
    >>> print(confusion_matrix(y_test, y_pred))
    [[ 23   0]
     [  2 225]]
    """

    def __init__(
        self,
        n_estimators=10,
        base_estimator=None,
        warm_start=False,
        sampling_strategy="auto",
        replacement=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
    ):
        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=False,
            bootstrap_features=False,
            oob_score=False,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        self.sampling_strategy = sampling_strategy
        self.replacement = replacement

    def _validate_estimator(self, default=AdaBoostClassifier()):
        """Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute."""
        if not isinstance(self.n_estimators, (numbers.Integral, np.integer)):
            raise ValueError(
                "n_estimators must be an integer, "
                "got {}.".format(type(self.n_estimators))
            )

        if self.n_estimators <= 0:
            raise ValueError(
                "n_estimators must be greater than zero, "
                "got {}.".format(self.n_estimators)
            )

        if self.base_estimator is not None:
            base_estimator = clone(self.base_estimator)
        else:
            base_estimator = clone(default)

        self.base_estimator_ = Pipeline(
            [
                (
                    "sampler",
                    RandomUnderSampler(
                        sampling_strategy=self.sampling_strategy,
                        replacement=self.replacement,
                    ),
                ),
                ("classifier", base_estimator),
            ]
        )

    def fit(self, X, y):
        """Train the ensemble on the training set.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,)
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        check_target_type(y)
        # RandomUnderSampler is not supporting sample_weight. We need to pass
        # None.
        return self._fit(X, y, self.max_samples, sample_weight=None)
