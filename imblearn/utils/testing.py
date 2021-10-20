"""Test utilities."""

# Adapted from scikit-learn
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import inspect
import pkgutil
import warnings
from contextlib import contextmanager
from importlib import import_module
from re import compile
from pathlib import Path

from operator import itemgetter
from pytest import warns as _warns

from sklearn.base import BaseEstimator
from sklearn.utils._testing import ignore_warnings


def all_estimators(
    type_filter=None,
):
    """Get a list of all estimators from imblearn.

    This function crawls the module and gets all classes that inherit
    from BaseEstimator. Classes that are defined in test-modules are not
    included.
    By default meta_estimators are also not included.
    This function is adapted from sklearn.

    Parameters
    ----------
    type_filter : str, list of str, or None, default=None
        Which kind of estimators should be returned. If None, no
        filter is applied and all estimators are returned.  Possible
        values are 'sampler' to get estimators only of these specific
        types, or a list of these to get the estimators that fit at
        least one of the types.

    Returns
    -------
    estimators : list of tuples
        List of (name, class), where ``name`` is the class name as string
        and ``class`` is the actual type of the class.
    """
    from ..base import SamplerMixin

    def is_abstract(c):
        if not (hasattr(c, "__abstractmethods__")):
            return False
        if not len(c.__abstractmethods__):
            return False
        return True

    all_classes = []
    modules_to_ignore = {"tests"}
    root = str(Path(__file__).parent.parent)
    # Ignore deprecation warnings triggered at import time and from walking
    # packages
    with ignore_warnings(category=FutureWarning):
        for importer, modname, ispkg in pkgutil.walk_packages(
            path=[root], prefix="imblearn."
        ):
            mod_parts = modname.split(".")
            if any(part in modules_to_ignore for part in mod_parts) or "._" in modname:
                continue
            module = import_module(modname)
            classes = inspect.getmembers(module, inspect.isclass)
            classes = [
                (name, est_cls) for name, est_cls in classes if not name.startswith("_")
            ]

            all_classes.extend(classes)

    all_classes = set(all_classes)

    estimators = [
        c
        for c in all_classes
        if (issubclass(c[1], BaseEstimator) and c[0] != "BaseEstimator")
    ]
    # get rid of abstract base classes
    estimators = [c for c in estimators if not is_abstract(c[1])]

    # get rid of sklearn estimators which have been imported in some classes
    estimators = [c for c in estimators if "sklearn" not in c[1].__module__]

    if type_filter is not None:
        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)  # copy
        filtered_estimators = []
        filters = {"sampler": SamplerMixin}
        for name, mixin in filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered_estimators.extend(
                    [est for est in estimators if issubclass(est[1], mixin)]
                )
        estimators = filtered_estimators
        if type_filter:
            raise ValueError(
                "Parameter type_filter must be 'sampler' or "
                "None, got"
                " %s." % repr(type_filter)
            )

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(estimators), key=itemgetter(0))


@contextmanager
def warns(expected_warning, match=None):
    r"""Assert that a warning is raised with an optional matching pattern.

    .. deprecated:: 0.8
       This function is deprecated in 0.8 and will be removed in 0.10.
       Use `pytest.warns()` instead.

    Assert that a code block/function call warns ``expected_warning``
    and raise a failure exception otherwise. It can be used within a context
    manager ``with``.

    Parameters
    ----------
    expected_warning : Warning
        Warning type.

    match : regex str or None, optional
        The pattern to be matched. By default, no check is done.

    Yields
    ------
    Nothing.

    Examples
    --------
    >>> import warnings
    >>> from imblearn.utils.testing import warns
    >>> with warns(UserWarning, match=r'must be \d+$'):
    ...     warnings.warn("value must be 42", UserWarning)
    """
    warnings.warn(
        "The warns function is deprecated in 0.8 and will be removed in 0.10. "
        "Use pytest.warns() instead."
    )

    with _warns(expected_warning) as record:
        yield

    if match is not None:
        for each in record:
            if compile(match).search(str(each.message)) is not None:
                break
        else:
            msg = "'{}' pattern not found in {}".format(
                match, "{}".format([str(r.message) for r in record])
            )
            assert False, msg
    else:
        pass
