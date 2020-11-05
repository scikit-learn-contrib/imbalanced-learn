import numpy as np
import pytest
from dask import array
from dask_ml.datasets import make_classification

from imblearn.dask.utils import is_multilabel
from imblearn.dask.utils import type_of_target


def test_type_of_target_error():
    y = np.arange(10)

    err_msg = "Expected a Dask array, series or dataframe."
    with pytest.raises(ValueError, match=err_msg):
        type_of_target(y)


@pytest.mark.parametrize(
    "y, expected_result",
    [
        (array.from_array(np.array([0, 1, 0, 1])), False),
        (array.from_array(np.array([[1, 0], [0, 0]])), True),
        (array.from_array(np.array([[1], [0], [0]])), False),
        (array.from_array(np.array([[1, 0, 0]])), True),
    ]
)
def test_is_multilabel(y, expected_result):
    assert is_multilabel(y) is expected_result


@pytest.mark.parametrize(
    "y, expected_type_of_target",
    [
        (array.from_array(np.array([[1, 0], [0, 0]])), "multilabel-indicator"),
        (array.from_array(np.array([[1, 0, 0]])), "multilabel-indicator"),
        (array.from_array(np.array([[[1, 2]]])), "unknown"),
        (array.from_array(np.array([[]])), "unknown"),
        (array.from_array(np.array([.1, .2, 3])), "continuous"),
        (array.from_array(np.array([[.1, .2, 3]])), "continuous-multioutput"),
        (array.from_array(np.array([[1., .2]])), "continuous-multioutput"),
        (array.from_array(np.array([1, 2])), "binary"),
        (array.from_array(np.array(["a", "b"])), "binary"),
    ]
)
def test_type_of_target(y, expected_type_of_target):
    target_type = type_of_target(y)
    assert target_type == expected_type_of_target
