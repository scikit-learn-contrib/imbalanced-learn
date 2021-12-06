from sklearn.externals._packaging.version import parse as parse_version


def to_numpy(obj):
    import pandas as pd

    pd_version = parse_version(pd.__version__)
    if pd_version >= parse_version("0.25.0"):
        return obj.to_numpy()
    else:
        return obj.values
