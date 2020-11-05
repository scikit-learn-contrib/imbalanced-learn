_REGISTERED_DASK_CONTAINER = []

try:
    from dask import array, dataframe
    _REGISTERED_DASK_CONTAINER += [
        array.Array, dataframe.Series, dataframe.DataFrame,
    ]
except ImportError:
    pass


def is_dask_container(container):
    return isinstance(container, tuple(_REGISTERED_DASK_CONTAINER))
