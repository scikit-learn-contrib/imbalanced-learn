"""Bagging classifier trained on balanced bootstrap samples."""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
#          Christos Aridas
# License: MIT

import numbers

import numpy as np

from sklearn.base import clone
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

from ..pipeline import Pipeline
from ..under_sampling import RandomUnderSampler
from ..under_sampling.base import BaseUnderSampler
from ..utils import Substitution, check_target_type, check_sampling_strategy
from ..utils._docstring import _n_jobs_docstring
from ..utils._docstring import _random_state_docstring
from ..utils._validation import _deprecate_positional_args


@Substitution(
    sampling_strategy=BaseUnderSampler._sampling_strategy_docstring,
    n_jobs=_n_jobs_docstring,
    random_state=_random_state_docstring,
)
class BalancedBaggingClassifier(BaggingClassifier):
    """A Bagging classifier with additional balancing.

    This implementation of Bagging is similar to the scikit-learn
    implementation. It includes an additional step to balance the training set
    at fit time using a given sampler.

    This classifier can serves as a basis to implement various methods such as
    Exactly Balanced Bagging [6]_, Roughly Balanced Bagging [7]_,
    Over-Bagging [6]_, or SMOTE-Bagging [8]_.

    Read more in the :ref:`User Guide <bagging>`.

    Parameters
    ----------
    base_estimator : estimator object, default=None
        The base estimator to fit on random subsets of the dataset.
        If None, then the base estimator is a decision tree.

    n_estimators : int, default=10
        The number of base estimators in the ensemble.

    max_samples : int or float, default=1.0
        The number of samples to draw from X to train each base estimator.

        - If int, then draw ``max_samples`` samples.
        - If float, then draw ``max_samples * X.shape[0]`` samples.

    max_features : int or float, default=1.0
        The number of features to draw from X to train each base estimator.

        - If int, then draw ``max_features`` features.
        - If float, then draw ``max_features * X.shape[1]`` features.

    bootstrap : bool, default=True
        Whether samples are drawn with replacement.

        .. note::
           Note that this bootstrap will be generated from the resampled
           dataset.

    bootstrap_features : bool, default=False
        Whether features are drawn with replacement.

    oob_score : bool, default=False
        Whether to use out-of-bag samples to estimate
        the generalization error.

    warm_start : bool, default=False
        When set to True, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit
        a whole new ensemble.

    {sampling_strategy}

    replacement : bool, default=False
        Whether or not to randomly sample with replacement or not when
        `sampler is None`, corresponding to a
        :class:`~imblearn.under_sampling.RandomUnderSampler`.

    {n_jobs}

    {random_state}

    verbose : int, default=0
        Controls the verbosity of the building process.

    sampler : sampler object, default=None
        The sampler used to balanced the dataset before to bootstrap
        (if `bootstrap=True`) and `fit` a base estimator. By default, a
        :class:`~imblearn.under_sampling.RandomUnderSampler` is used.

        .. versionadded:: 0.8

    Attributes
    ----------
    base_estimator_ : estimator
        The base estimator from which the ensemble is grown.

    n_features_ : int
        The number of features when `fit` is performed.

    estimators_ : list of estimators
        The collection of fitted base estimators.

    estimators_samples_ : list of ndarray
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator. Each subset is defined by a boolean mask.

    estimators_features_ : list of ndarray
        The subset of drawn features for each base estimator.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_classes_ : int or list
        The number of classes.

    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.

    oob_decision_function_ : ndarray of shape (n_samples, n_classes)
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        ``oob_decision_function_`` might contain NaN.

    See Also
    --------
    BalancedRandomForestClassifier : Random forest applying random-under
        sampling to balance the different bootstraps.

    EasyEnsembleClassifier : Ensemble of AdaBoost classifier trained on
        balanced bootstraps.

    RUSBoostClassifier : AdaBoost classifier were each bootstrap is balanced
        using random-under sampling at each round of boosting.

    Notes
    -----
    This is possible to turn this classifier into a balanced random forest [5]_
    by passing a :class:`~sklearn.tree.DecisionTreeClassifier` with
    `max_features='auto'` as a base estimator.

    See
    :ref:`sphx_glr_auto_examples_ensemble_plot_comparison_ensemble_classifier.py`.

    References
    ----------
    .. [1] L. Breiman, "Pasting small votes for classification in large
           databases and on-line", Machine Learning, 36(1), 85-103, 1999.

    .. [2] L. Breiman, "Bagging predictors", Machine Learning, 24(2), 123-140,
           1996.

    .. [3] T. Ho, "The random subspace method for constructing decision
           forests", Pattern Analysis and Machine Intelligence, 20(8), 832-844,
           1998.

    .. [4] G. Louppe and P. Geurts, "Ensembles on Random Patches", Machine
           Learning and Knowledge Discovery in Databases, 346-361, 2012.

    .. [5] C. Chen Chao, A. Liaw, and L. Breiman. "Using random forest to
           learn imbalanced data." University of California, Berkeley 110,
           2004.

    .. [6] R. Maclin, and D. Opitz. "An empirical evaluation of bagging and
           boosting." AAAI/IAAI 1997 (1997): 546-551.

    .. [7] S. Hido, H. Kashima, and Y. Takahashi. "Roughly balanced bagging
           for imbalanced data." Statistical Analysis and Data Mining: The ASA
           Data Science Journal 2.5‐6 (2009): 412-426.

    .. [8] S. Wang, and X. Yao. "Diversity analysis on imbalanced data sets by
           using ensemble models." 2009 IEEE symposium on computational
           intelligence and data mining. IEEE, 2009.

    Examples
    --------
    >>> from collections import Counter
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import confusion_matrix
    >>> from imblearn.ensemble import \
BalancedBaggingClassifier # doctest: +NORMALIZE_WHITESPACE
    >>> X, y = make_classification(n_classes=2, class_sep=2,
    ... weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0,
    ... n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)
    >>> print('Original dataset shape %s' % Counter(y))
    Original dataset shape Counter({{1: 900, 0: 100}})
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     random_state=0)
    >>> bbc = BalancedBaggingClassifier(random_state=42)
    >>> bbc.fit(X_train, y_train) # doctest: +ELLIPSIS
    BalancedBaggingClassifier(...)
    >>> y_pred = bbc.predict(X_test)
    >>> print(confusion_matrix(y_test, y_pred))
    [[ 23   0]
     [  2 225]]
    """

    @_deprecate_positional_args
    def __init__(
        self,
        base_estimator=None,
        n_estimators=10,
        *,
        max_samples=1.0,
        max_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        oob_score=False,
        warm_start=False,
        sampling_strategy="auto",
        replacement=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        sampler=None,
    ):

        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )
        self.sampling_strategy = sampling_strategy
        self.replacement = replacement
        self.sampler = sampler

    def _validate_y(self, y):
        y_encoded = super()._validate_y(y)
        if (
            isinstance(self.sampling_strategy, dict)
            and self.sampler_._sampling_type != "bypass"
        ):
            self._sampling_strategy = {
                np.where(self.classes_ == key)[0][0]: value
                for key, value in check_sampling_strategy(
                    self.sampling_strategy,
                    y,
                    self.sampler_._sampling_type,
                ).items()
            }
        else:
            self._sampling_strategy = self.sampling_strategy
        return y_encoded

    def _validate_estimator(self, default=DecisionTreeClassifier()):
        """Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute."""
        if not isinstance(self.n_estimators, (numbers.Integral, np.integer)):
            raise ValueError(
                f"n_estimators must be an integer, " f"got {type(self.n_estimators)}."
            )

        if self.n_estimators <= 0:
            raise ValueError(
                f"n_estimators must be greater than zero, " f"got {self.n_estimators}."
            )

        if self.base_estimator is not None:
            base_estimator = clone(self.base_estimator)
        else:
            base_estimator = clone(default)

        if self.sampler_._sampling_type != "bypass":
            self.sampler_.set_params(sampling_strategy=self._sampling_strategy)

        self.base_estimator_ = Pipeline(
            [
                ("sampler", self.sampler_),
                ("classifier", base_estimator),
            ]
        )

    def fit(self, X, y):
        """Build a Bagging ensemble of estimators from the training set (X, y).

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
        # the sampler needs to be validated before to call _fit because
        # _validate_y is called before _validate_estimator and would require
        # to know which type of sampler we are using.
        if self.sampler is None:
            self.sampler_ = RandomUnderSampler(
                replacement=self.replacement,
            )
        else:
            self.sampler_ = clone(self.sampler)
        # RandomUnderSampler is not supporting sample_weight. We need to pass
        # None.
        return self._fit(X, y, self.max_samples, sample_weight=None)

    def _more_tags(self):
        tags = super()._more_tags()
        tags_key = "_xfail_checks"
        failing_test = "check_estimators_nan_inf"
        reason = "Fails because the sampler removed infinity and NaN values"
        if tags_key in tags:
            tags[tags_key][failing_test] = reason
        else:
            tags[tags_key] = {failing_test: reason}
        return tags
