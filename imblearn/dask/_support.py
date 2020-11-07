def is_dask_collection(container):
    try:
        # to keep dask as an optional depency, keep the statement in a
        # try/except statement
        from dask import is_dask_collection

        return is_dask_collection(container)
    except ImportError:
        return False
