"""Test utilities."""

# Adapted from scikit-learn
# Authors: Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: MIT

import inspect
import pkgutil
from contextlib import contextmanager
from re import compile

from operator import itemgetter
from pytest import warns as _warns

from sklearn.base import BaseEstimator

from imblearn.base import SamplerMixin
import imblearn

# meta-estimators need another estimator to be instantiated.
META_ESTIMATORS = []
# estimators that there is no way to default-construct sensibly
OTHER = ["Pipeline", "FeatureUnion", "SMOTENC"]
# some strange ones
DONT_TEST = []


def all_estimators(include_meta_estimators=False,
                   include_other=False,
                   type_filter=None,
                   include_dont_test=False):
    """Get a list of all estimators from imblearn.

    This function crawls the module and gets all classes that inherit
    from BaseEstimator. Classes that are defined in test-modules are not
    included.
    By default meta_estimators are also not included.
    This function is adapted from sklearn.

    Parameters
    ----------
    include_meta_estimators : boolean, default=False
        Whether to include meta-estimators that can be constructed using
        an estimator as their first argument. These are currently none.

    include_other : boolean, default=False
        Wether to include meta-estimators that are somehow special and can
        not be default-constructed sensibly. These are currently
        Pipeline, FeatureUnion.

    include_dont_test : boolean, default=False
        Whether to include "special" label estimator or test processors.

    type_filter : string, list of string, or None, default=None
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

    def is_abstract(c):
        if not (hasattr(c, '__abstractmethods__')):
            return False
        if not len(c.__abstractmethods__):
            return False
        return True

    all_classes = []
    # get parent folder
    path = imblearn.__path__
    for importer, modname, ispkg in pkgutil.walk_packages(
            path=path, prefix='imblearn.', onerror=lambda x: None):
        if (".tests." in modname):
            continue
        module = __import__(modname, fromlist="dummy")
        classes = inspect.getmembers(module, inspect.isclass)
        all_classes.extend(classes)

    all_classes = set(all_classes)

    estimators = [
        c for c in all_classes
        if (issubclass(c[1], BaseEstimator) and c[0] != 'BaseEstimator')
    ]
    # get rid of abstract base classes
    estimators = [c for c in estimators if not is_abstract(c[1])]

    # get rid of sklearn estimators which have been imported in some classes
    estimators = [c for c in estimators if "sklearn" not in c[1].__module__]

    if not include_dont_test:
        estimators = [c for c in estimators if not c[0] in DONT_TEST]

    if not include_other:
        estimators = [c for c in estimators if not c[0] in OTHER]
    # possibly get rid of meta estimators
    if not include_meta_estimators:
        estimators = [c for c in estimators if not c[0] in META_ESTIMATORS]
    if type_filter is not None:
        if not isinstance(type_filter, list):
            type_filter = [type_filter]
        else:
            type_filter = list(type_filter)  # copy
        filtered_estimators = []
        filters = {'sampler': SamplerMixin}
        for name, mixin in filters.items():
            if name in type_filter:
                type_filter.remove(name)
                filtered_estimators.extend(
                    [est for est in estimators if issubclass(est[1], mixin)])
        estimators = filtered_estimators
        if type_filter:
            raise ValueError("Parameter type_filter must be 'sampler' or "
                             "None, got"
                             " %s." % repr(type_filter))

    # drop duplicates, sort for reproducibility
    # itemgetter is used to ensure the sort does not extend to the 2nd item of
    # the tuple
    return sorted(set(estimators), key=itemgetter(0))


@contextmanager
def warns(expected_warning, match=None):
    """Assert that a warning is raised with an optional matching pattern

    Assert that a code block/function call warns ``expected_warning``
    and raise a failure exception otherwise. It can be used within a context
    manager ``with``.

    Parameters
    ----------
    expected_warning : Warning
        Warning type.

    match : regex str or None, optional
        The pattern to be matched. By default, no check is done.

    Returns
    -------
    None

    Examples
    --------

    >>> import warnings
    >>> from imblearn.utils.testing import warns
    >>> with warns(UserWarning, match=r'must be \d+$'):
    ...     warnings.warn("value must be 42", UserWarning)

    """
    with _warns(expected_warning) as record:
        yield

    if match is not None:
        for each in record:
            if compile(match).search(str(each.message)) is not None:
                break
        else:
            msg = "'{}' pattern not found in {}".format(
                match, '{}'.format([str(r.message) for r in record]))
            assert False, msg
    else:
        pass
