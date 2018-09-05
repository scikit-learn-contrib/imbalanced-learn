"""Utilities for deprecation"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import warnings


def deprecate_parameter(sampler,
                        version_deprecation,
                        param_deprecated,
                        new_param=None):
    """Helper to deprecate a parameter by another one.

    Parameters
    ----------
    sampler : object,
        The object which will be inspected.

    version_deprecation : str,
        The version from which the parameter will be deprecated. The format
        should be ``'x.y'``

    param_deprecated : str,
        The parameter being deprecated.

    new_param : str,
        The parameter used instead of the deprecated parameter. By default, no
        parameter is expected.

    Returns
    -------
    None

    """
    x, y = version_deprecation.split('.')
    version_removed = x + '.' + str(int(y) + 2)
    if new_param is None:
        if getattr(sampler, param_deprecated) is not None:
            warnings.warn(
                "'{}' is deprecated from {} and will be removed in"
                " {} for the estimator {}."
                .format(param_deprecated, version_deprecation,
                        version_removed, sampler.__class__),
                category=DeprecationWarning)
    else:
        if getattr(sampler, param_deprecated) is not None:
            warnings.warn(
                "'{}' is deprecated from {} and will be removed in"
                " {} for the estimator {}. Use '{}' instead."
                .format(param_deprecated, version_deprecation,
                        version_removed, sampler.__class__, new_param),
                category=DeprecationWarning)
            setattr(sampler, new_param, getattr(sampler, param_deprecated))
