import numpy as np

from sklearn.utils.multiclass import check_classification_targets as \
    sklearn_check_classification_targets
from sklearn.utils.multiclass import type_of_target as sklearn_type_of_target
from sklearn.utils.validation import column_or_1d as sklearn_column_or_1d

from ..dask._support import is_dask_container


def type_of_target(y):
    if is_dask_container(y):
        from ..dask.utils import type_of_target as dask_type_of_target

        return dask_type_of_target(y)
    return sklearn_type_of_target(y)


def _is_multiclass_encoded(y):
    if is_dask_container(y):
        from dask import array

        return array.all(y.sum(axis=1) == 1).compute()
    return np.all(y.sum(axis=1) == 1)


def column_or_1d(y, *, warn=False):
    if is_dask_container(y):
        from ..dask.utils import column_or_1d as dask_column_or_1d

        return dask_column_or_1d(y, warn=warn)
    return sklearn_column_or_1d(y, warn=warn)


def unique(arr, **kwargs):
    if is_dask_container(arr):
        if hasattr(arr, "unique"):
            output = np.asarray(arr.unique(**kwargs))
        else:
            output = np.unique(arr).compute()
        return output
    return np.unique(arr, **kwargs)


def check_classification_targets(y):
    if is_dask_container(y):
        from ..dask.utils import check_classification_targets as \
            dask_check_classification_targets

        return dask_check_classification_targets(y)
    return sklearn_check_classification_targets(y)
