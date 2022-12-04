from numbers import Integral, Real

from ..utils._param_validation import HasMethods, Interval, StrOptions


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


_bagging_parameter_constraints = {
    "estimator": [HasMethods(["fit", "predict"]), None],
    "n_estimators": [Interval(Integral, 1, None, closed="left")],
    "max_samples": [
        Interval(Integral, 1, None, closed="left"),
        Interval(Real, 0, 1, closed="right"),
    ],
    "max_features": [
        Interval(Integral, 1, None, closed="left"),
        Interval(Real, 0, 1, closed="right"),
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
