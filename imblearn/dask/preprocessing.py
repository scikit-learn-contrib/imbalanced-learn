import numpy as np


def label_binarize(y, *, classes):
    import pandas as pd
    from dask import dataframe

    cat_dtype = pd.CategoricalDtype(categories=classes)
    y = dataframe.from_array(y).astype(cat_dtype)
    return dataframe.get_dummies(y).to_dask_array()
