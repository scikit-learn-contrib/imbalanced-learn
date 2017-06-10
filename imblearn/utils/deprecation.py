"""Utilities for deprecation"""

# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import warnings


def deprecate_parameter(sampler, version_deprecation, param_deprecated,
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
    warnings.simplefilter("always", DeprecationWarning)
    if new_param is None:
        if getattr(sampler, param_deprecated) is not None:
            warnings.warn("'{}' is deprecated from {} and will be removed in"
                          " {}.".format(param_deprecated,
                                        version_deprecation,
                                        str(float(version_deprecation) + 0.2)),
                          category=DeprecationWarning)
    else:
        if getattr(sampler, param_deprecated) is not None:
            warnings.warn("'{}' is deprecated from {} and will be removed in"
                          " {}. Use '{}' instead.".format(
                              param_deprecated,
                              version_deprecation,
                              str(float(version_deprecation) + 0.2),
                              new_param),
                          category=DeprecationWarning)
            setattr(sampler, new_param, getattr(sampler, param_deprecated))
