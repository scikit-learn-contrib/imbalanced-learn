"""Compatibility fixes for older version of python, numpy, scipy, and
scikit-learn.

If you add content to this file, please give the version of the package at
which the fix is no longer needed.
"""
import functools
import sys

import scipy
import scipy.stats
import sklearn
from sklearn.utils.fixes import parse_version

from .._config import config_context, get_config

sp_version = parse_version(scipy.__version__)
sklearn_version = parse_version(parse_version(sklearn.__version__).base_version)


# TODO: Remove when SciPy 1.9 is the minimum supported version
def _mode(a, axis=0):
    if sp_version >= parse_version("1.9.0"):
        return scipy.stats.mode(a, axis=axis, keepdims=True)
    return scipy.stats.mode(a, axis=axis)


# TODO: remove when scikit-learn minimum version is 1.3
if sklearn_version < parse_version("1.3"):

    def _fit_context(*, prefer_skip_nested_validation):
        """Decorator to run the fit methods of estimators within context managers.

        Parameters
        ----------
        prefer_skip_nested_validation : bool
            If True, the validation of parameters of inner estimators or functions
            called during fit will be skipped.

            This is useful to avoid validating many times the parameters passed by the
            user from the public facing API. It's also useful to avoid validating
            parameters that we pass internally to inner functions that are guaranteed to
            be valid by the test suite.

            It should be set to True for most estimators, except for those that receive
            non-validated objects as parameters, such as meta-estimators that are given
            estimator objects.

        Returns
        -------
        decorated_fit : method
            The decorated fit method.
        """

        def decorator(fit_method):
            @functools.wraps(fit_method)
            def wrapper(estimator, *args, **kwargs):
                global_skip_validation = get_config()["skip_parameter_validation"]

                # we don't want to validate again for each call to partial_fit
                partial_fit_and_fitted = (
                    fit_method.__name__ == "partial_fit" and _is_fitted(estimator)
                )

                if not global_skip_validation and not partial_fit_and_fitted:
                    estimator._validate_params()

                with config_context(
                    skip_parameter_validation=(
                        prefer_skip_nested_validation or global_skip_validation
                    )
                ):
                    return fit_method(estimator, *args, **kwargs)

            return wrapper

        return decorator

else:
    from sklearn.base import _fit_context  # type: ignore[no-redef] # noqa

# TODO: remove when scikit-learn minimum version is 1.3
if sklearn_version < parse_version("1.3"):

    def _is_fitted(estimator, attributes=None, all_or_any=all):
        """Determine if an estimator is fitted

        Parameters
        ----------
        estimator : estimator instance
            Estimator instance for which the check is performed.

        attributes : str, list or tuple of str, default=None
            Attribute name(s) given as string or a list/tuple of strings
            Eg.: ``["coef_", "estimator_", ...], "coef_"``

            If `None`, `estimator` is considered fitted if there exist an
            attribute that ends with a underscore and does not start with double
            underscore.

        all_or_any : callable, {all, any}, default=all
            Specify whether all or any of the given attributes must exist.

        Returns
        -------
        fitted : bool
            Whether the estimator is fitted.
        """
        if attributes is not None:
            if not isinstance(attributes, (list, tuple)):
                attributes = [attributes]
            return all_or_any([hasattr(estimator, attr) for attr in attributes])

        if hasattr(estimator, "__sklearn_is_fitted__"):
            return estimator.__sklearn_is_fitted__()

        fitted_attrs = [
            v for v in vars(estimator) if v.endswith("_") and not v.startswith("__")
        ]
        return len(fitted_attrs) > 0

else:
    from sklearn.utils.validation import _is_fitted  # type: ignore[no-redef]

try:
    from sklearn.utils.validation import _is_pandas_df
except ImportError:

    def _is_pandas_df(X):
        """Return True if the X is a pandas dataframe."""
        if hasattr(X, "columns") and hasattr(X, "iloc"):
            # Likely a pandas DataFrame, we explicitly check the type to confirm.
            try:
                pd = sys.modules["pandas"]
            except KeyError:
                return False
            return isinstance(X, pd.DataFrame)
        return False


if sklearn_version < parse_version("1.6"):
    from sklearn.utils._tags import _safe_tags as get_tags
else:
    from sklearn.utils import get_tags

if sklearn_version < parse_version("1.6"):
    def validate_data(_estimator, **kwargs):
        if "ensure_all_finite" in kwargs:
            force_all_finite = kwargs.pop("ensure_all_finite")
        else:
            force_all_finite = True
        return _estimator._validate_data(**kwargs, force_all_finite=force_all_finite)
else:
    from sklearn.utils.validation import validate_data  # type: ignore[no-redef]


if sklearn_version < parse_version("1.6"):
    def _check_n_features(estimator, X, *, reset):
        return estimator._check_n_features(X, reset=reset)
else:
    from sklearn.utils.validation import _check_n_features  # type: ignore[no-redef]

if sklearn_version < parse_version("1.6"):
    def _check_feature_names(estimator, X, *, reset):
        return estimator._check_feature_names(X, reset=reset)
else:
    from sklearn.utils.validation import _check_feature_names  # type: ignore[no-redef]


def check_version_package(package, constraint, version, /):
    """Create a function to check package version against a constraint.

    Parameters
    ----------
    package : str
        The package name to check version for.
    constraint : {"<", "<=", ">", ">="}
        The version constraint.
    version : str
        The version to compare against.

    Returns
    -------
    callable
        A function that takes an estimator and returns bool.
    """
    operators = {
        "<": lambda x, y: x < y,
        "<=": lambda x, y: x <= y,
        ">": lambda x, y: x > y,
        ">=": lambda x, y: x >= y
    }

    if constraint not in operators:
        raise ValueError(f"Invalid constraint: {constraint}")

    op = operators[constraint]
    parsed_version = parse_version(version)

    def check_version(estimator):
        try:
            pkg = __import__(package)
            pkg_version = parse_version(parse_version(pkg.__version__).base_version)
            return op(pkg_version, parsed_version)
        except (ImportError, AttributeError):
            return False

    return check_version


# TODO: Remove when python>=3.10 is the minimum supported version
def _dataclass_args():
    if sys.version_info < (3, 10):
        return {}
    return {"slots": True}
