import copy
from numbers import Integral, Real

import numpy as np
from sklearn.tree._criterion import Criterion

from ..utils._param_validation import (
    HasMethods,
    Hidden,
    Interval,
    RealNotInt,
    StrOptions,
)


def _estimator_has(attr):
    """Check if we can delegate a method to the underlying estimator.
    First, we check the first fitted estimator if available, otherwise we
    check the estimator attribute.
    """

    def check(self):
        if hasattr(self, "estimators_"):
            return hasattr(self.estimators_[0], attr)
        elif self.estimator is not None:
            return hasattr(self.estimator, attr)
        else:  # TODO(1.4): Remove when the base_estimator deprecation cycle ends
            return hasattr(self.base_estimator, attr)

    return check


def _estimate_reweighting(estimators):
    """Estimate the reweighting factor to calibrate the probabilities.

    The reweighting factor is the averaged ratio of the probability of the
    positive class before and after resampling for all samplers.

    Parameters
    ----------
    estimators : list of estimators
        The list of fitted estimators. Each estimator is a
        :class:`~imblearn.pipeline.Pipeline` where the first stage is a sampler.

    Returns
    -------
    weight : float
        The reweighting factor.
    """
    weights = []
    for estimator in estimators:
        sampler = estimator[0]
        # Since the samplers are internally created, we know that we have target encoded
        # with 0 and 1.
        p_y_1_original = sampler._original_class_counts[1] / sum(
            sampler._original_class_counts[k] for k in [0, 1]
        )
        resampled_counts = copy.copy(sampler._original_class_counts)
        resampled_counts.update(sampler.sampling_strategy_)
        p_y_1_resampled = resampled_counts[1] / sum(resampled_counts[k] for k in [0, 1])
        weights.append(
            (p_y_1_original / (1 - p_y_1_original))
            * ((1 - p_y_1_resampled) / p_y_1_resampled)
        )
    return np.mean(weights)


_bagging_parameter_constraints = {
    "estimator": [HasMethods(["fit", "predict"]), None],
    "n_estimators": [Interval(Integral, 1, None, closed="left")],
    "max_samples": [
        Interval(Integral, 1, None, closed="left"),
        Interval(RealNotInt, 0, 1, closed="right"),
    ],
    "max_features": [
        Interval(Integral, 1, None, closed="left"),
        Interval(RealNotInt, 0, 1, closed="right"),
    ],
    "bootstrap": ["boolean"],
    "bootstrap_features": ["boolean"],
    "oob_score": ["boolean"],
    "warm_start": ["boolean"],
    "n_jobs": [None, Integral],
    "random_state": ["random_state"],
    "verbose": ["verbose"],
    "base_estimator": [
        HasMethods(["fit", "predict"]),
        StrOptions({"deprecated"}),
        None,
    ],
}

_adaboost_classifier_parameter_constraints = {
    "estimator": [HasMethods(["fit", "predict"]), None],
    "n_estimators": [Interval(Integral, 1, None, closed="left")],
    "learning_rate": [Interval(Real, 0, None, closed="neither")],
    "random_state": ["random_state"],
    "base_estimator": [HasMethods(["fit", "predict"]), StrOptions({"deprecated"})],
    "algorithm": [StrOptions({"SAMME", "SAMME.R"})],
}

_random_forest_classifier_parameter_constraints = {
    "n_estimators": [Interval(Integral, 1, None, closed="left")],
    "bootstrap": ["boolean"],
    "oob_score": ["boolean"],
    "n_jobs": [Integral, None],
    "random_state": ["random_state"],
    "verbose": ["verbose"],
    "warm_start": ["boolean"],
    "criterion": [StrOptions({"gini", "entropy", "log_loss"}), Hidden(Criterion)],
    "max_samples": [
        None,
        Interval(Real, 0.0, 1.0, closed="right"),
        Interval(Integral, 1, None, closed="left"),
    ],
    "max_depth": [Interval(Integral, 1, None, closed="left"), None],
    "min_samples_split": [
        Interval(Integral, 2, None, closed="left"),
        Interval(RealNotInt, 0.0, 1.0, closed="right"),
    ],
    "min_samples_leaf": [
        Interval(Integral, 1, None, closed="left"),
        Interval(RealNotInt, 0.0, 1.0, closed="neither"),
    ],
    "min_weight_fraction_leaf": [Interval(Real, 0.0, 0.5, closed="both")],
    "max_features": [
        Interval(Integral, 1, None, closed="left"),
        Interval(RealNotInt, 0.0, 1.0, closed="right"),
        StrOptions({"sqrt", "log2"}),
        None,
    ],
    "max_leaf_nodes": [Interval(Integral, 2, None, closed="left"), None],
    "min_impurity_decrease": [Interval(Real, 0.0, None, closed="left")],
    "ccp_alpha": [Interval(Real, 0.0, None, closed="left")],
    "class_weight": [
        StrOptions({"balanced_subsample", "balanced"}),
        dict,
        list,
        None,
    ],
    "monotonic_cst": ["array-like", None],
}
