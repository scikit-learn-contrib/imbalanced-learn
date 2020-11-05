import warnings

from dask import array
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.multiclass import _is_integral_float


def is_multilabel(y):
    if not (y.ndim == 2 and y.shape[1] > 1):
        return False

    labels = array.unique(y).compute()

    return len(labels) < 3 and (
        y.dtype.kind in 'biu' or _is_integral_float(labels)
    )


def type_of_target(y):
    if is_multilabel(y):
        return 'multilabel-indicator'

    if y.ndim > 2:
        return 'unknown'

    if y.ndim == 2 and y.shape[1] == 0:
        return 'unknown'  # [[]]

    if y.ndim == 2 and y.shape[1] > 1:
        # [[1, 2], [1, 2]]
        suffix = "-multioutput"
    else:
        # [1, 2, 3] or [[1], [2], [3]]
        suffix = ""

    # check float and contains non-integer float values
    if y.dtype.kind == 'f' and array.any(y != y.astype(int)):
        # [.1, .2, 3] or [[.1, .2, 3]] or [[1., .2]] and not [1., 2., 3.]
        # NOTE: we don't check for infinite values
        return 'continuous' + suffix

    labels = array.unique(y).compute()
    if (len((labels)) > 2) or (y.ndim >= 2 and len(y[0]) > 1):
        # [1, 2, 3] or [[1., 2., 3]] or [[1, 2]]
        return 'multiclass' + suffix
    # [1, 2] or [["a"], ["b"]]
    return 'binary'


def column_or_1d(y, *, warn=False):
    shape = y.shape
    if len(shape) == 1:
        return y.ravel()
    if len(shape) == 2 and shape[1] == 1:
        if warn:
            warnings.warn(
                "A column-vector y was passed when a 1d array was  expected. "
                "Please change the shape of y to (n_samples, ), for example "
                "using ravel().", DataConversionWarning, stacklevel=2
            )
        return y.ravel()

    raise ValueError(
        f"y should be a 1d array. Got an array of shape {shape} instead."
    )
